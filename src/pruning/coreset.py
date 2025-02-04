import torch


class CoresetSelection:
    @staticmethod
    def score_monotonic_selection(
        data_score, original_data, key="score_key", ratio=0.5, descending=True
    ):
        score = data_score[key]

        if score.dim() != 1:
            raise ValueError(
                f"Expected score tensor to be 1D, but got shape {score.shape}"
            )

        score_sorted_index = score.argsort(descending=descending)
        total_num = int(ratio * score.shape[0])
        selected_indices = score_sorted_index[:total_num]

        # Use selected indices to select corresponding data
        selected_data = original_data[selected_indices]

        return selected_indices, selected_data

    @staticmethod
    def stratified_sampling(
        data_score, original_data, coreset_key="score_key", coreset_num=None
    ):
        if coreset_num is None:
            # Default to 50% sampling if not specified
            coreset_num = int(0.5 * original_data.shape[0])

        stratas = 50
        score = data_score[coreset_key]

        if score.dim() == 0:
            score = score.unsqueeze(0)

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_indices = []
        strata_sizes = []

        for i in range(stratas):
            start, end = bin_range(i)
            mask = (score >= start) & (score < end)
            indices = torch.where(mask)[0]
            strata_indices.append(indices)
            strata_sizes.append(indices.numel())

        total_samples = sum(strata_sizes)
        if total_samples == 0:
            print("No data to sample.")
            return torch.tensor([], dtype=torch.long), torch.tensor(
                [], dtype=torch.float
            )

        budgets = [
            max(int(s / total_samples * coreset_num), 1) if s > 0 else 0
            for s in strata_sizes
        ]

        total_budget = sum(budgets)
        if total_budget > coreset_num:
            scaling_factor = coreset_num / total_budget
            budgets = [int(b * scaling_factor) for b in budgets]

        selected_indices = []
        for indices, budget in zip(strata_indices, budgets):
            if indices.numel() > 0 and budget > 0:
                if indices.numel() <= budget:
                    selected = indices
                else:
                    rand_indices = torch.randperm(indices.numel())[:budget]
                    selected = indices[rand_indices]
                selected_indices.append(selected)

        if selected_indices:
            selected_indices = torch.cat(selected_indices)
            if selected_indices.numel() > coreset_num:
                selected_indices = selected_indices[:coreset_num]
        else:
            selected_indices = torch.tensor([], dtype=torch.long)

        # Use selected indices to select corresponding data
        selected_data = original_data[selected_indices]

        return selected_indices, selected_data
