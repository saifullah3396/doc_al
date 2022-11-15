from __future__ import annotations

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy


@register_strategy("random_sampling")
class RandomSampling(QueryStrategy):
    def query(self, n_samples: int):
        import torch

        random_unlabled_indices = torch.randperm(len(self._datamodule.unlabeled_indices))[:n_samples]
        return self._datamodule.unlabeled_indices[random_unlabled_indices]
