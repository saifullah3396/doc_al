"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple, Union

from al.core.args import ActiveLearningArguments
from al.core.training.query_strategies.factory import QueryStrategyFactory
from xai_torch.core.args import Arguments
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.trainer import Trainer
from xai_torch.core.training.utilities import reset_random_seeds
from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

if TYPE_CHECKING:
    import torch
    from al.core.training.query_strategies.base import QueryStrategy
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from ignite.engine import Engine
    from pathlib import Path

from xai_torch.core.training.constants import TrainingStage
from xai_torch.core.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)


class DALTrainer(Trainer):
    @staticmethod
    def get_trainer(al_args: ActiveLearningArguments):
        if al_args.query_strategy == "vaal":
            from .vaal_trainer import VAALTrainer

            return VAALTrainer
        elif al_args.query_strategy == "waal":
            from al.core.training.waal_trainer import WAALTrainer

            return WAALTrainer
        elif al_args.query_strategy == "lpl":
            from .lpl_trainer import LPLTrainer

            return LPLTrainer
        else:
            return DALTrainer

    @classmethod
    def setup_datamodule(
        cls, args: Arguments, rank: int = 0, stage: TrainingStage = TrainingStage.train
    ) -> BaseDataModule:
        """
        Initializes the datamodule for training.
        """
        import ignite.distributed as idist
        from al.core.data.active_learning_datamodule import ActiveLearningDataModule
        from xai_torch.core.data.data_modules.factory import DataModuleFactory
        from xai_torch.core.models.factory import ModelFactory

        # get data collator required for the model
        model_class = ModelFactory.get_model_class(args.model_args)
        collate_fns = model_class.get_data_collators(args)

        # initialize data module generator function
        datamodule = DataModuleFactory.create(args=args, collate_fns=collate_fns)

        # only download dataset on rank 0, all other ranks wait here for rank 0 to load the datasets
        if rank > 0:
            idist.barrier()

        # wrap the datamodule with DALDataModule to generate labeled/unlabeled samples
        if stage == TrainingStage.train or stage is None:
            datamodule = ActiveLearningDataModule(datamodule, args.al_args)

        # we manually prepare data and call setup here so dataset related properties can be initalized.
        datamodule.prepare_data(quiet=False, stage=stage)
        datamodule.setup(quiet=False, stage=stage)

        if rank == 0:
            idist.barrier()

        # Todo: Fix with new code
        # call prepare data on start so that initial variables are set correctly
        # datamodule.prepare_data()
        # if args.general_args.debug_data:
        #     datamodule.prepare_data()
        #     datamodule.setup()
        #     for stage in list(TrainingStage):
        #         if stage == TrainingStage.predict:
        #             continue
        #         logger.info(f"Visualizing data batch for training stage = [{stage}]")
        #         image_grid = datamodule.show_batch(stage=stage, show=False)
        #         for pl_logger in pl_loggers:
        #             if isinstance(pl_logger, TensorBoardLogger):
        #                 writer = pl_logger.experiment
        #                 writer.add_image(f"Images for stage = {stage.value}", image_grid)

        return datamodule

    @classmethod
    def initialize_prediction_engine(
        cls,
        model: XAIModel,
        device: Optional[Union[str, torch.device]],
        allow_dropout: bool = False,
    ) -> Callable:
        from ignite.engine import Engine

        def prediction_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            # ready model for evaluation
            if allow_dropout:
                model.torch_model.train()
            else:
                model.torch_model.eval()
            with torch.no_grad():
                # put batch to device
                batch = convert_tensor(batch, device=device)

                # forward pass
                return model.torch_model.predict_step(batch)

        return Engine(prediction_step)

    @classmethod
    def save_round_state(
        cls,
        curr_round: int,
        datamodule: ActiveLearningDataModule,
        output_dir: Union[str, Path],
    ):
        from ignite.handlers.checkpoint import DiskSaver

        al_state = {
            "curr_round": curr_round,
            "datamodule": {
                "labeled_indices_mask": datamodule.labeled_indices_mask,
                "labeled_indices": datamodule.labeled_indices,
                "unlabeled_indices": datamodule.unlabeled_indices,
            },
        }
        if datamodule.pseudo_label_indices is not None:
            al_state["datamodule"]["pseudo_label_indices"] = datamodule.pseudo_label_indices
        if datamodule.pseudo_labels is not None:
            al_state["datamodule"]["pseudo_labels"] = datamodule.pseudo_labels

        checkpoint_dir = output_dir
        save_handler = DiskSaver(
            checkpoint_dir,
            require_empty=False,
        )
        save_handler(al_state, "al_state.pth")
        save_handler(al_state, f"al_state_{al_state['curr_round']}.pth")

    @classmethod
    def load_round_state(cls, curr_round: int, datamodule: ActiveLearningDataModule, output_dir: Union[str, Path]):
        import logging

        import torch

        al_state = {
            "curr_round": curr_round,
            "datamodule": {
                "labeled_indices_mask": datamodule.labeled_indices_mask,
                "labeled_indices": datamodule.labeled_indices,
                "unlabeled_indices": datamodule.unlabeled_indices,
            },
        }

        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        state_path = output_dir / "al_state.pth"
        if state_path.exists():
            logger.info(f"Resuming training from the active learning checkpoint {str(state_path)}")
            state = torch.load(state_path)
            for k, v in state.items():
                al_state[k] = v
            datamodule.load_state(al_state["datamodule"])
        return al_state

    @classmethod
    def train(cls, local_rank, args: Arguments):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        import ignite.distributed as idist
        from xai_torch.core.training.utilities import initialize_training, setup_logging
        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME, setup_logger

        # setup logging
        logger = setup_logger(DEFAULT_LOGGER_NAME, distributed_rank=local_rank, level=logging.INFO)

        # initialize training
        initialize_training(args)

        # initialize torch device (cpu or gpu)
        device = idist.device()

        # get device rank
        rank = idist.get_rank()

        # initialize logging directory and tensorboard logger
        output_dir, tb_logger = setup_logging(args)

        # setup datamodule
        datamodule: ActiveLearningDataModule = cls.setup_datamodule(args, rank=rank, stage=None)

        # setup model
        model = cls.setup_model(args, datamodule, tb_logger, summarize=True)

        # define active learning query strategy
        query_strategy: QueryStrategy = QueryStrategyFactory.create(
            datamodule=datamodule, model=model, device=device, args=args.al_args
        )

        # load active learning state
        al_state = DALTrainer.load_round_state(0, datamodule, output_dir=output_dir)
        curr_round = al_state["curr_round"]

        if curr_round == args.al_args.n_rounds:
            logger.warning(
                "Active learning rounds have already been finished! Either increase the number of "
                f"max rounds (current={args.al_args.n_rounds}) "
                "OR reset the training from start."
            )
            exit()

        # reset seeds for training. This allows multiple experiments with same seed for dataset initialization but
        # different seeds for the active learning training process.
        reset_random_seeds(args.al_args.al_seed)

        while curr_round < args.al_args.n_rounds:
            from al.core.training.query_strategies.impl.ceal import CEAL

            logger.info(f"============== Running round={curr_round} of active learning ===========")
            # update tblogger dir
            tb_logger = None
            if rank == 0:
                from ignite.contrib.handlers import TensorboardLogger

                tb_logger = TensorboardLogger(output_dir / str(curr_round))

            # print labels summary
            datamodule.print_label_summary()

            # Reset model for re-training
            if args.al_args.reset_model:
                model = cls.setup_model(args, datamodule, tb_logger, summarize=False)
            else:
                # Reset only optimizers and schedulers
                model._opt_sch_handler.setup_opt_sch()

            # get train dataloader for labelled data
            if isinstance(query_strategy, CEAL) and curr_round > 0 and datamodule._pseudo_labeled_dataset is not None:
                labeled_dataloader = datamodule.pseudo_labeled_dataloader()
            else:
                labeled_dataloader = datamodule.labeled_dataloader()

            # get validation data loader
            val_dataloader = datamodule.val_dataloader()

            # setup training engine
            training_engine, _ = cls.setup_training_engine(
                args=args,
                model=model,
                train_dataloader=labeled_dataloader,
                val_dataloader=val_dataloader,
                output_dir=output_dir / str(curr_round),  # append round number to output_dir
                tb_logger=tb_logger,
                device=device,
            )
            training_engine.logger = logger

            # NOTE: The test engine has already updated the model state with state of last/best
            # checkpoint which will be used for querying of the next round.
            def perform_query():
                import timeit

                # reset the querying strategy
                query_strategy.reset(model)

                # update the labeled pool
                start = timeit.default_timer()
                n_query_samples = int(args.al_args.n_query_ratio * datamodule.pool_size)
                if isinstance(query_strategy, CEAL):
                    query_indices = query_strategy.query(n_samples=n_query_samples, round=curr_round)
                else:
                    query_indices = query_strategy.query(n_samples=n_query_samples)
                stop = timeit.default_timer()
                tb_logger.writer.add_scalar("query_time", stop - start, curr_round)
                datamodule.update_dataset_labels(query_indices)

            def test_model():
                # after the training, the test engine automatically loads the 'best' model to continue the rounds.
                test_dataloader = datamodule.test_dataloader()

                # run testing after the end of every round
                test_engine = cls.setup_test_engine(
                    args=args,
                    model=model,
                    test_dataloader=test_dataloader,
                    output_dir=output_dir / str(curr_round),
                    tb_logger=tb_logger,
                    device=device,
                )
                test_engine.logger = logger
                test_engine.run(test_dataloader)

            resume_epoch = training_engine.state.epoch
            if not (training_engine._is_done(training_engine.state) or resume_epoch >= args.training_args.max_epochs):
                # run training
                training_engine.run(labeled_dataloader, max_epochs=args.training_args.max_epochs)

                # perform query
                perform_query()

                # test model
                test_model()
            else:
                # if we are resuming from last checkpoint and training is already finished
                logger.info(
                    "Training has already been finished! Either increase the number of "
                    f"epochs (current={args.training_args.max_epochs}) >= {resume_epoch} "
                    "OR reset the training from start."
                )

                # perform query
                perform_query()

                # test model
                test_model()

            # save active learning query state for next round
            DALTrainer.save_round_state(curr_round + 1, datamodule, output_dir=output_dir)

            if rank == 0:
                # close tb logger
                tb_logger.close()

            curr_round += 1
