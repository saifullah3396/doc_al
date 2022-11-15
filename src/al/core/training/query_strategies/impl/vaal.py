from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from xai_torch.core.constants import DataKeys

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from al.core.models.vaal.xai_model import VAALXAIModel
    from ignite.engine import Engine


# class VAALTrainer:
#     @classmethod
#     def configure_running_avg_logging(cls, engine: Engine):
#         from functools import partial

#         from ignite.metrics import RunningAverage

#         def output_transform(x: Any, index: int, name: str) -> Any:
#             import numbers

#             import torch

#             if isinstance(x, Mapping):
#                 return x[name]
#             elif isinstance(x, Sequence):
#                 return x[index]
#             elif isinstance(x, (torch.Tensor, numbers.Number)):
#                 return x
#             else:
#                 raise TypeError(
#                     "Unhandled type of update_function's output. "
#                     f"It should either mapping or sequence, but given {type(x)}"
#                 )

#         # add loss as a running average metric
#         for i, n in enumerate(["total_vae_loss", "dsc_loss"]):
#             RunningAverage(output_transform=partial(output_transform, index=i, name=n), epoch_bound=False).attach(
#                 engine, f"vaal/{n}"
#             )


def initialize_vaal_prediction_engine(
    model: VAALXAIModel, device: Optional[Union[str, torch.device]] = torch.device("cpu")
) -> Callable:
    from ignite.engine import Engine

    def step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        """
        Define the step per batch for deep fool
        """

        from ignite.utils import convert_tensor

        # ready model for evaluation
        model.eval()

        # put batch to device
        batch = convert_tensor(batch, device=device)

        # forward pass
        _, _, mu, _ = model.torch_model.vae(batch[DataKeys.IMAGE])
        scores = model.torch_model.dsc(mu).cpu().view(-1)

        # return
        return {"scores": scores}

    return Engine(step)


@register_strategy("vaal")
class VAAL(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: VAALXAIModel,
        device: torch.device,
        embedding_layer: str = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self.setup_vaal_engine()

    def reset(self, model):
        super().reset(model)
        self.setup_vaal_engine()

    def setup_vaal_engine(self):
        from al.core.training.metrics.output_gatherer import OutputGatherer
        from ignite.contrib.handlers import ProgressBar
        from torch import optim

        # setup the vaal training engine
        self._vaal_prediction_engine = initialize_vaal_prediction_engine(
            self._model,
            device=self._device,
        )

        # attach output accumulator
        epoch_metric = OutputGatherer(output_transform=lambda output: output["scores"])
        epoch_metric.attach(self._vaal_prediction_engine, "scores")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._vaal_prediction_engine, metric_names="all")

    def get_pred_scores(self):
        dataloader = self._datamodule.unlabeled_dataloader(batch_size=self._batch_size)
        return self._vaal_prediction_engine.run(dataloader).metrics["scores"]

    def query(self, n_samples: int):
        uncertainties = self.get_pred_scores()
        return self._datamodule.unlabeled_indices[uncertainties.sort(descending=True)[1][:n_samples]]
