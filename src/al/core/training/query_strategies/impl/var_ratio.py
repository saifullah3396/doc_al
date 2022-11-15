from __future__ import annotations

from typing import TYPE_CHECKING

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel


@register_strategy("var_ratio")
class VarRatio(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)

    def query(self, n_samples: int):
        import torch
        probs = self.get_probs_on_unlabeled()
        preds = torch.max(probs, 1)[0]
        uncertainties = 1.0 - preds
        return self._datamodule.unlabeled_indices[uncertainties.sort(descending=True)[1][:n_samples]]

