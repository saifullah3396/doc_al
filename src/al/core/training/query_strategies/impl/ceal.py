from __future__ import annotations

from typing import TYPE_CHECKING

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

from .entropy_sampling import EntropySampling

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel


@register_strategy("ceal")
class CEAL(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        dropout_runs: int = 10,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._entropy_sampling = EntropySampling(datamodule, model, device, embedding_layer)
        self._dropout_runs = dropout_runs
        self.delta = 5 * 1e-5

    def reset(self, model):
        super().reset(model)
        self._entropy_sampling = EntropySampling(self._datamodule, self._model, self._device, self._embedding_layer)
        self.setup_prediction_engines()

    def query(self, n_samples: int, round: int):
        import copy

        import numpy as np

        query_indices = self._entropy_sampling.query(n_samples)
        self.delta = self.delta - 0.033 * 1e-5 * round
        self._datamodule.update_dataset_labels(query_indices)

        probs = self.get_probs_on_unlabeled().numpy()
        preds = probs.argmax(-1)

        entropy = (-1.0 * probs * np.log(probs)).sum(1)
        high_confident_idx = np.where(entropy < self.delta, True, False)

        pseudo_label_indices = self._datamodule.unlabeled_indices[high_confident_idx]
        self._datamodule.prepare_pseudo_labels_dataset(pseudo_label_indices, preds[high_confident_idx])
        return query_indices
