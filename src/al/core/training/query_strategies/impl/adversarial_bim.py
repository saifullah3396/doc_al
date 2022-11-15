from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple, Union

import torch
from xai_torch.core.constants import DataKeys

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

if TYPE_CHECKING:
    import torch
    from xai_torch.core.models.xai_model import XAIModel

    from al.core.data.active_learning_datamodule import ActiveLearningDataModule


def adv_bim(model, batch, eps: float = 0.05, max_iterations=100):
    import torch.nn.functional as F

    image = batch[DataKeys.IMAGE]
    image.requires_grad_()
    eta = torch.zeros(image.shape, device=image.device)

    logits = model.predict_step({DataKeys.IMAGE: image + eta})[DataKeys.LOGITS]
    py = logits.max(1)[1]
    ny = logits.max(1)[1]

    for i in range(max_iterations):
        batch_mask = py == ny
        if torch.all(batch_mask == False):
            # break the loop of all images have their attacks successful
            break

        loss = F.cross_entropy(logits, ny)
        loss.backward()

        eta[batch_mask] += eps * torch.sign(image.grad.data)[batch_mask]
        image.grad.data.zero_()

        logits = model.predict_step({DataKeys.IMAGE: image + eta})[DataKeys.LOGITS]
        py = logits.max(1)[1]

    return (eta * eta).sum(dim=(1, 2, 3))


def initialize_adv_bim_engine(
    model: XAIModel, device: Optional[Union[str, torch.device]] = torch.device("cpu"), eps: float = 0.05
) -> Callable:
    from ignite.engine import Engine

    def step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        """
        Define the step per batch for deep fool
        """

        from ignite.utils import convert_tensor

        # ready model for evaluation
        model.torch_model.eval()

        # put batch to device
        batch = convert_tensor(batch, device=device)

        # forward pass
        return {"adv_dis": adv_bim(model.torch_model, batch, eps=eps)}

    return Engine(step)


@register_strategy("adv_bim")
class AdversarialBIM(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        eps: float = 0.05,
        batch_size: int = 16,
        unlabeled_subset_size: int = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._eps = eps
        self._batch_size = batch_size
        self._unlabeled_subset_size = unlabeled_subset_size

        self.setup_adv_engine()

    def reset(self, model):
        super().reset(model)

        self.setup_adv_engine()

    def setup_adv_engine(self):
        from ignite.contrib.handlers import ProgressBar

        from al.core.training.metrics.output_gatherer import OutputGatherer

        # setup the adversarial engine to apply adversarial attacks on the unlabelled dataset
        self._adversarial_engine = initialize_adv_bim_engine(model=self._model, device=self._device, eps=self._eps)

        # attach output accumulator
        epoch_metric = OutputGatherer(output_transform=lambda output: output["adv_dis"])
        epoch_metric.attach(self._adversarial_engine, "adv_dis")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._adversarial_engine)

    def get_adv_dis(self):
        if self._unlabeled_subset_size is not None:
            dataloader, self.unlabeled_subset = self._datamodule.unlabeled_dataloader(
                batch_size=self._batch_size, unlabeled_subset_size=self._unlabeled_subset_size
            )
        else:
            dataloader = self._datamodule.unlabeled_dataloader(batch_size=self._batch_size)
        return self._adversarial_engine.run(dataloader).metrics["adv_dis"]

    def query(self, n_samples: int):
        adv_dis = self.get_adv_dis()
        if self._unlabeled_subset_size is not None:
            return torch.tensor(self.unlabeled_subset.indices)[adv_dis.argsort()[:n_samples]]
        else:
            return self._datamodule.unlabeled_indices[adv_dis.argsort()[:n_samples]]
