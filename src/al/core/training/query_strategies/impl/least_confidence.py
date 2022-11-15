from __future__ import annotations

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy


@register_strategy("least_confidence")
class LeastConfidence(QueryStrategy):
    def query(self, n_samples: int):
        uncertainties = self.get_probs_on_unlabeled().max(1)[0]
        return self._datamodule.unlabeled_indices[uncertainties.sort()[1][:n_samples]]
