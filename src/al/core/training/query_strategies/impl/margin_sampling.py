from __future__ import annotations

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy


@register_strategy("margin_sampling")
class MarginSampling(QueryStrategy):
    def query(self, n_samples: int):
        probs_sorted, idxs = self.get_probs_on_unlabeled().sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        return self._datamodule.unlabeled_indices[uncertainties.sort()[1][:n_samples]]
