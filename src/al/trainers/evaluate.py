"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import hydra
from omegaconf import DictConfig
from xai_torch.trainers.evaluate import run


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    app()
