import torch
from timm.models.vision_transformer import VisionTransformer, Block, Attention
from torch import nn
from typing import Tuple
from patch_merging.merge import bipartite_soft_matching, merge_source, merge_wavg
from patch_merging.utils import parse_r

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks.
     - Compute and propagate token size and potentially the token sources.
    """
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            merge, _ = bipartite_soft_matching(
                metric, r, self._tome_info["class_token"], self._tome_info["distill_token"]
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Adding a linear layer for token processing (transforming feature_size if necessary)
            self.token_embed = nn.Linear(self.embed_dim, self.embed_dim)  # Adjust `embed_dim` as needed

        def forward(self, x: torch.Tensor, *args, **kwdargs) -> torch.Tensor:
            """
            Override the forward pass to accept tokenized input directly (shape: [batch_size, num_tokens, feature_size]).
            """
            # If the input tensor has shape [num_tokens, feature_size], add batch dimension
            if len(x.shape) == 2:  # Shape [num_tokens, feature_size]
                x = x.unsqueeze(0)  # Add batch dimension: shape becomes [batch_size, num_tokens, feature_size]

            # Apply a linear transformation to the token features (if necessary)
            x = self.token_embed(x)

            # Skip the patch embedding and positional encoding, as the tokens are already embedded
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            # Apply positional embedding (optional)
            x = self.pos_embed(x)

            # Pass the tokens through transformer blocks
            x = self.blocks(x)

            # Final normalization
            x = self.norm(x)
            return x

    return ToMeVisionTransformer


def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
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

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
