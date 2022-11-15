from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing import Event
from typing import TYPE_CHECKING

from al.core.training.metrics.output_gatherer import OutputGatherer
from al.core.training.query_strategies.utilities import logits_output_transform
from al.core.training.trainer import DALTrainer

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel


class QueryStrategy(ABC):
    def __init__(
        self, datamodule: ActiveLearningDataModule, model: XAIModel, device: torch.device, embedding_layer: str = None
    ):
        self._datamodule = datamodule
        self._model = model
        self._device = device
        self._embedding_layer = embedding_layer

        self.setup_prediction_engines()

    @abstractmethod
    def query(self, n_samples: int):
        pass

    def reset(self, model):
        self._model = model
        self.setup_prediction_engines()

    def train(self):
        pass

    def setup_prediction_engines(self):
        from ignite.contrib.handlers import ProgressBar

        # setup the prediction engine to get output logits
        self._prediction_engine = DALTrainer.initialize_prediction_engine(model=self._model, device=self._device)
        self._prediction_engine_with_dropout = DALTrainer.initialize_prediction_engine(
            model=self._model, device=self._device, allow_dropout=True
        )

        epoch_metric = OutputGatherer(output_transform=logits_output_transform)
        epoch_metric.attach(self._prediction_engine, "logits")

        epoch_metric = OutputGatherer(output_transform=logits_output_transform)
        epoch_metric.attach(self._prediction_engine_with_dropout, "logits")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._prediction_engine)

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._prediction_engine_with_dropout)

    def attach_embedding_hook(self, embeddings):
        def accumulate_embeddings(module, input, output):
            embeddings.append(output.detach().cpu().squeeze())

        self._embedding_hook = None
        for k, v in self._model.torch_model.named_modules():
            if k == self._embedding_layer:
                self._embedding_hook = v.register_forward_hook(accumulate_embeddings)

        if self._embedding_hook is None:
            raise ValueError("Requested embedding layer does not exist in the model.")

    def detach_embedding_hook(self):
        self._embedding_hook.remove()

    def get_logits_on_unlabeled(self):
        return self._prediction_engine.run(self._datamodule.unlabeled_dataloader()).metrics["logits"].cpu()

    def get_preds_on_unlabeled(self):
        return self.get_logits_on_unlabeled().argmax(-1)

    def get_probs_on_unlabeled(self):
        import torch.nn.functional as F

        return F.softmax(self.get_logits_on_unlabeled(), dim=1)

    def get_probs_on_unlabeled_with_dropout(self, dropout_runs=10, take_mean: bool = True):
        import torch
        import torch.nn.functional as F
        import tqdm

        probs_list = []
        for _ in tqdm.tqdm(range(dropout_runs)):
            output = self._prediction_engine_with_dropout.run(self._datamodule.unlabeled_dataloader())
            output = F.softmax(output.metrics["logits"].cpu(), dim=1)
            probs_list.append(output)
        probs_list = torch.stack(probs_list)
        if take_mean:
            return probs_list.mean(0)
        else:
            return probs_list

    def get_embeddings_from_unlabeled_data(self, return_logits=False):
        import torch

        embeddings = []
        self.attach_embedding_hook(embeddings)
        results = self._prediction_engine.run(self._datamodule.unlabeled_dataloader())
        self.detach_embedding_hook()

        embeddings = torch.cat(embeddings)
        if not return_logits:
            return embeddings
        return embeddings, results.metrics["logits"].cpu()
