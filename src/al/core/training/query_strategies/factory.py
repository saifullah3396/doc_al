"""
Defines the factory for DataAugmentation class and its children.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from al.core.args import ActiveLearningArguments
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel


class QueryStrategyFactory:
    @staticmethod
    def create(
        datamodule: ActiveLearningDataModule, model: XAIModel, device: torch.device, args: ActiveLearningArguments
    ):
        from al.core.training.query_strategies.constants import QUERY_STRATEGY_REGISTRY

        strategy_class = QUERY_STRATEGY_REGISTRY.get(args.query_strategy, None)
        if strategy_class is None:
            raise ValueError(f"Query strategy [{args.query_strategy}] is not supported.")
        return strategy_class(datamodule=datamodule, model=model, device=device, **args.query_strategy_kwargs)
