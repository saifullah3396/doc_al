from __future__ import annotations

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy


@register_strategy("entropy_sampling")
class EntropySampling(QueryStrategy):
    def query(self, n_samples: int):
        import torch

        probs = self.get_probs_on_unlabeled()
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)
        return self._datamodule.unlabeled_indices[uncertainties.sort()[1][:n_samples]]
