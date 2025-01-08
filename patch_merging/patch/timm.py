import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block, Attention, PatchEmbed
from typing import Optional, Tuple
from patch_merging.merge import bipartite_soft_matching, merge_source, merge_wavg
from patch_merging.utils import parse_r

class ToMePatchEmbed(PatchEmbed):
    """
    Custom Patch Embedding class to process pre-tokenized input (a list of tokens).
    Skips the patch embedding process and directly accepts tokenized input.
    """

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept a list of tokens directly without applying patch embedding.
        The input is assumed to be of shape [batch_size, num_tokens, feature_size].
        """
        x = self.norm(x)
        return x 

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # Compute and print the mean and standard deviation for x
        print("before processing ----------- ") 
        print(f'Mean of x: {x.mean().item()}')
        print(f'Std of x: {x.std().item()}')  
        
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        print(f'Mean of x: {x.mean().item()}')
        print(f'Std of x: {x.std().item()}')   
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        print(attn.shape)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)

def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace patch embedding with custom ToMePatchEmbed (doesn't do patching)
            # self.patch_embed = ToMePatchEmbed(embed_dim=self.embed_dim, img_size=224, patch_size=16)
            # self.patch_embed  = nn.Identity()
              
        def forward(self, x: torch.Tensor, *args, **kwdargs) -> torch.Tensor:
            """
            Override the forward pass to accept tokenized input directly.
            """
            if len(x.shape) == 2:  # If input has shape [num_tokens, feature_size], add batch dimension
                x = x.unsqueeze(0)
            
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(x, *args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
    model.pos_embed = None

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True
    
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
        elif isinstance(module, PatchEmbed):
            module.__class__ = ToMePatchEmbed
         
