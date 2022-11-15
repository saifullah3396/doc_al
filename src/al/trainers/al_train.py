"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
from al.core.args import ActiveLearningArguments
from al.core.training.trainer import DALTrainer
from omegaconf import DictConfig, OmegaConf
from xai_torch.core.args import Arguments
from xai_torch.utilities.dacite_wrapper import from_dict

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)


def run(args: Arguments_):
    if args.general_args.do_train:
        try:
            print(f"{'='*50} Starting training {'='*50}")
            DALTrainer.get_trainer(args.al_args).train(0, args)
            print(f"{'='*50} Training finished {'='*50}")
        except KeyboardInterrupt:
            logging.info("Received ctrl-c interrupt. Stopping training...")
        except Exception as e:
            logging.exception(e)
            exit(1)

    if args.general_args.do_test:
        try:
            print(f"\n\n{'='*50} Starting testing {'='*50}")
            DALTrainer.get_trainer(args.al_args).test(args)
            print(f"{'='*50} Testing finished {'='*50}")
        except Exception as e:
            logging.exception(e)


@dataclass
class Arguments_(Arguments):
    """
    Add our our arguments to base arguments class here
    """

    # Hydra will populate this field based on the defaults list
    al_args: ActiveLearningArguments = ActiveLearningArguments()


@hydra.main(version_base=None, config_path="../../cfg", config_name="hydra")
def app(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    args = from_dict(data_class=Arguments_, data=cfg["args"])
    run(args)


if __name__ == "__main__":
    app()
