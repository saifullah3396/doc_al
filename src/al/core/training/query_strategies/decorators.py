def register_strategy(reg_name: str = ""):
    from al.core.training.query_strategies.base import QueryStrategy
    from al.core.training.query_strategies.constants import QUERY_STRATEGY_REGISTRY
    from xai_torch.utilities.decorators import register_as_child

    return register_as_child(
        base_class_type=QueryStrategy,
        registry=QUERY_STRATEGY_REGISTRY,
        reg_name=reg_name,
    )
