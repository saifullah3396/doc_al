"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from dataclasses import dataclass, field
from typing import Mapping, Optional

from xai_torch.core.args_base import ArgumentsBase


@dataclass
class ActiveLearningArguments(ArgumentsBase):
    query_strategy: str = "random_sampling"
    query_strategy_kwargs: dict = field(
        default_factory=lambda: {},
    )
    n_query_ratio: float = 0.1
    labeled_split_ratio: float = 0.2
    n_rounds: int = 10
    embedding_layer: str = ""
    val_samples_ratio: Optional[float] = 0.2
    class_imbalance: bool = False
    n_classes_removed: int = 2
    reset_model: bool = False
    al_seed: int = 0
    training_args: dict = field(
        default_factory=lambda: {},
    )

    def __post_init__(self):
        from xai_torch.utilities.dacite_wrapper import from_dict

        if self.query_strategy == "vaal":
            self.training_args = from_dict(
                data_class=VAALTrainingArguments,
                data=self.training_args,
            )
        elif self.query_strategy == "lpl":
            self.training_args = from_dict(
                data_class=LPLTrainingArguments,
                data=self.training_args,
            )
        elif self.query_strategy == "waal":
            self.training_args = from_dict(
                data_class=WAALTrainingArguments,
                data=self.training_args,
            )


@dataclass
class VAALTrainingArguments:
    num_vae_steps: int = 2
    num_adv_steps: int = 1
    beta: int = 1
    adv_param: int = 1
    latent_dim: int = 32
    vae: str = "default"
    dsc: str = "default"


@dataclass
class LPLTrainingArguments:
    loss_model: str = "cifar10"
    loss_backprop_epochs: int = 30
    margin: float = 1.0
    weight: float = 1.0


@dataclass
class WAALTrainingArguments:
    gamma_ratio: float = 1.0
    alpha: float = 1e-3
