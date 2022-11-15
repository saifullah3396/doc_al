"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch
from al.core.data.collators import JointBatchToTensorDataCollator
from al.core.models.vaal.xai_model import VAALXAIModel
from al.core.training.query_strategies.factory import QueryStrategyFactory
from al.core.training.trainer import DALTrainer
from ignite.contrib.handlers import TensorboardLogger
from torch import nn
from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.utilities.data_collators import BatchToTensorDataCollator
from xai_torch.core.models.xai_model import XAIModel
from xai_torch.core.training.utilities import reset_random_seeds

if TYPE_CHECKING:
    from al.core.training.query_strategies.base import QueryStrategy
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from ignite.engine import Engine

from xai_torch.core.training.constants import TrainingStage

logging.basicConfig(level=logging.INFO)


class VAALTrainer(DALTrainer):
    @classmethod
    def configure_running_avg_logging(cls, args: Arguments, engine: Engine, stage: TrainingStage):
        from ignite.metrics import RunningAverage

        def output_transform(x: Any, index: int, name: str) -> Any:
            import numbers

            import torch

            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, (torch.Tensor, numbers.Number)):
                return x
            else:
                raise TypeError(
                    "Unhandled type of update_function's output. "
                    f"It should either mapping or sequence, but given {type(x)}"
                )

        # add loss as a running average metric
        for i, n in enumerate([f"{step}_{DataKeys.LOSS}" for step in ["task", "vae", "dsc"]]):
            RunningAverage(
                alpha=0.5, output_transform=partial(output_transform, index=i, name=n), epoch_bound=False
            ).attach(engine, f"{stage}/{n}")

    @classmethod
    def setup_training_engine(cls, args, model, train_dataloader, val_dataloader, output_dir, tb_logger, device):
        # setup training engine
        training_engine = cls.initialize_training_engine(
            args=args, model=model, train_dataloader=train_dataloader, device=device
        )

        validation_engine = None
        if args.general_args.do_val:
            # setup validation engine
            validation_engine = cls.initialize_validation_engine(args=args, model=model, device=device)

        # configure training and validation engines
        cls.configure_training_engine(
            args=args,
            training_engine=training_engine,
            model=model,
            output_dir=output_dir,
            tb_logger=tb_logger,
            train_dataloader=train_dataloader,
            validation_engine=validation_engine,
            val_dataloader=val_dataloader,
        )

        # add training hooks from the model
        model.add_training_hooks(training_engine)

        return training_engine, validation_engine

    @classmethod
    def initialize_training_engine(
        cls,
        args: Arguments,
        model: VAALXAIModel,
        train_dataloader: DataLoader,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    ) -> Callable:
        def cycle(iterable):
            while True:
                for i in iterable:
                    yield i

        if args.training_args.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

        from ignite.engine import Engine

        # get related arguments
        gradient_accumulation_steps = args.training_args.gradient_accumulation_steps
        non_blocking = args.training_args.non_blocking_tensor_conv
        train_datacycler = cycle(train_dataloader)

        def update_model(engine, model, batch, step="task"):
            from xai_torch.core.constants import DataKeys

            # perform optimizers zero_grad() operation with gradient accumulation
            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                print(step, "zero grad")
                model.optimizers[step].zero_grad()

            # forward pass
            model_output = model.torch_model.training_step(batch=batch, step=step)

            # make sure we get a dict from the model
            assert isinstance(model_output, dict), "Model must return an instance of dict."

            # get loss from the output dict
            loss = model_output[DataKeys.LOSS]

            # accumulate loss if required
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # backward pass
            loss.backward()
            print(step, loss)

            # perform optimizer update for correct gradient accumulation step
            if engine.state.iteration % gradient_accumulation_steps == 0:
                model.optimizers[step].step()
                print(step, "step update")

            # if on the go training evaluation is required, detach data from the graph
            if args.training_args.eval_training and step == "task":
                return_dict = {}
                for key, value in model_output.items():
                    if key == DataKeys.LOSS:
                        return_dict[key] = value.item()
                    elif isinstance(value, torch.Tensor):
                        return_dict[key] = value.detach()
                return return_dict

            return {f"{step}_{DataKeys.LOSS}": model_output[DataKeys.LOSS].item()}

        def training_step(engine: Engine, _) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            from ignite.utils import convert_tensor

            # setup model for training
            model.torch_model.train()

            # get batch from dataloader
            batch = next(train_datacycler)

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            # call task model update
            task_output = update_model(engine, model, batch, step="task")

            # call the vae update
            for count in range(args.al_args.training_args.num_vae_steps):
                vae_output = update_model(engine, model, batch, step="vae")

                # sample new batch if needed to train the adversarial network
                if count < (args.al_args.training_args.num_vae_steps - 1):
                    batch = next(train_datacycler)
                    batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            # call the dsc update
            for count in range(args.al_args.training_args.num_adv_steps):
                dsc_output = update_model(engine, model, batch, step="dsc")

                # sample new batch if needed to train the adversarial network
                if count < (args.al_args.training_args.num_adv_steps - 1):
                    batch = next(train_datacycler)
                    batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            return {**task_output, **vae_output, **dsc_output}

        return Engine(training_step)

    @classmethod
    def setup_model(
        cls,
        args: Arguments,
        datamodule: BaseDataModule,
        tb_logger: TensorboardLogger,
        summarize: bool = False,
        stage: TrainingStage = TrainingStage.train,
    ) -> XAIModel:
        """
        Initializes the model for training.
        """
        from xai_torch.core.models.factory import ModelFactory

        # setup model
        model = ModelFactory.create(args, datamodule, tb_logger=tb_logger, wrapper_class=VAALXAIModel)
        model.setup(stage=stage)

        # generate model summary
        if summarize:
            model.summarize()

        return model

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
            joint_dataloader = datamodule.get_joint_dataset_loader(
                collate_fn=JointBatchToTensorDataCollator(datamodule._collate_fns.train)
            )

            # get validation data loader
            val_dataloader = datamodule.val_dataloader()

            # setup training engine
            training_engine, _ = cls.setup_training_engine(
                args=args,
                model=model,
                train_dataloader=joint_dataloader,
                val_dataloader=val_dataloader,
                output_dir=output_dir / str(curr_round),  # append round number to output_dir
                tb_logger=tb_logger,
                device=device,
            )
            training_engine.logger = logger

            resume_epoch = training_engine.state.epoch
            if not (training_engine._is_done(training_engine.state) or resume_epoch >= args.training_args.max_epochs):
                # run training
                training_engine.run(range(len(joint_dataloader)), max_epochs=args.training_args.max_epochs)
                # training_engine.run(labeled_dataloader, max_epochs=args.training_args.max_epochs)

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
            else:
                # if we are resuming from last checkpoint and training is already finished
                logger.info(
                    "Training has already been finished! Either increase the number of "
                    f"epochs (current={args.training_args.max_epochs}) >= {resume_epoch} "
                    "OR reset the training from start."
                )

                # after the training, the test engine automatically loads the 'best' model to continue the rounds.
                test_dataloader = datamodule.test_dataloader()

                # don't run test but just set it up so that model has latest correct checkpoint loaded
                test_engine = cls.setup_test_engine(
                    args=args,
                    model=model,
                    test_dataloader=test_dataloader,
                    output_dir=output_dir / str(curr_round),
                    tb_logger=tb_logger,
                    device=device,
                )

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

            # perform query
            perform_query()

            # save active learning query state for next round
            DALTrainer.save_round_state(curr_round + 1, datamodule, output_dir=output_dir)

            if rank == 0:
                # close tb logger
                tb_logger.close()

            curr_round += 1
