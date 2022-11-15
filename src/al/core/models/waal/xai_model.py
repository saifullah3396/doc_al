"""
WAAL base module for active learning.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from al.core.models.waal.base_module import WAALBaseModule
from ignite.contrib.handlers import TensorboardLogger
from xai_torch.core.models.base import BaseModule
from xai_torch.core.models.xai_model import OptimizersSchedulersHandler, XAIModel

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule

logging.basicConfig(level=logging.INFO)


class WAALXAIModel(XAIModel):
    def __init__(
        self,
        args: Arguments,
        model_class: BaseModule,
        datamodule: Optional[BaseDataModule] = None,
        tb_logger: Optional[TensorboardLogger] = None,
    ):
        super().__init__(args, model_class, datamodule, tb_logger)

        import ignite.distributed as idist

        # models sometimes download pretrained checkpoints when initializing. Only download it on rank 0
        if idist.get_rank() > 0:  # stop all ranks > 0
            idist.barrier()

        # wrap the model with WAAL model
        self._torch_model = WAALBaseModule(args, tb_logger, self._torch_model)

        # build model
        self._torch_model.build_model()

        # initialize metrics
        self._torch_model.init_metrics()

        # wait for rank 0 to download checkpoints
        if idist.get_rank() == 0:
            idist.barrier()

        # initialize optimizers schedulers handler
        self._opt_sch_handler = OptimizersSchedulersHandler(self._args, self)
