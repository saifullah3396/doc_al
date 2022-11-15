from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Callable, Optional

from torch.utils.data import DataLoader, Dataset, Subset
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.data_modules.base import BaseDataModule
from xai_torch.core.data.datasets.base import DatasetBase
from xai_torch.core.training.constants import TrainingStage

if TYPE_CHECKING:
    from al.train import ActiveLearningArguments


class JointDataset(Dataset):
    def __init__(self, labeled, unlabeled):
        self._labeled = labeled
        self._unlabeled = unlabeled

    def __len__(self):
        return max(len(self._labeled), len(self._unlabeled))

    def __getitem__(self, index):
        l1 = len(self._labeled)
        l2 = len(self._unlabeled)

        # checking the index in the range or not
        if index < l1:
            s1 = self._labeled[index]
        else:
            # rescaling the index to the range of Len1
            re_index = index % l1
            s1 = self._labeled[re_index]

        # checking second datasets
        if index < l2:
            s2 = self._unlabeled[index]
        else:
            # rescaling the index to the range of Len2
            re_index = index % l2
            s2 = self._unlabeled[re_index]
        return {"labeled": s1, "unlabeled": s2}


class ActiveLearningDataModule(BaseDataModule):
    def __init__(self, datamodule: BaseDataModule, dal_args: ActiveLearningArguments):
        self.__class__ = type(datamodule.__class__.__name__, (self.__class__, datamodule.__class__), {})
        self.__dict__ = datamodule.__dict__
        self._dal_args = dal_args

        # define mask for labeled/unlabeled indices
        self._labeled_indices_mask = None
        self._labeled_indices = None
        self._unlabeled_indices = None
        self._pool_size = None

        # init pseudo stuff
        self._pseudo_label_indices = None
        self._pseudo_labels = None
        self._pseudo_labeled_dataset = None

    @property
    def pool_size(self):
        return self._pool_size

    @property
    def labeled_indices_mask(self):
        return self._labeled_indices_mask

    @property
    def labeled_indices(self):
        return self._labeled_indices

    @property
    def unlabeled_indices(self):
        return self._unlabeled_indices

    @property
    def pseudo_label_indices(self):
        return self._pseudo_label_indices

    @property
    def pseudo_labels(self):
        return self._pseudo_labels

    def setup(self, quiet=False, stage: TrainingStage = TrainingStage.train) -> None:
        super().setup(quiet, stage)

        import copy

        import torch
        import tqdm
        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

        logger = getLogger(DEFAULT_LOGGER_NAME)

        logger.info("Setting up active learning datamodule.")

        # create labled and unlabeled datasets
        self._labeled_dataset = copy.deepcopy(self.train_dataset)
        self._unlabeled_dataset = copy.deepcopy(self.train_dataset)

        # get labeled data size
        self._pool_size = len(self.train_dataset)

        # get shuffled indices
        train_indices = torch.randperm(self._pool_size)

        # get labeled_dataset_size
        labeled_dataset_size = int(self._dal_args.labeled_split_ratio * self._pool_size)

        # initialize indices mask
        self._labeled_indices_mask = torch.zeros(self._pool_size, dtype=torch.bool)

        # generate imabalanced initial labeled set
        if self._dal_args.class_imbalance:
            # extract all labels from the dataset
            if isinstance(self.train_dataset, Subset):
                self.train_dataset.dataset._load_images = False
            else:
                self.train_dataset._load_images = False

            labels_per_sample = []
            for sample in tqdm.tqdm(self.train_dataset):
                labels_per_sample.append(sample[DataKeys.LABEL])
            labels_per_sample = torch.tensor(labels_per_sample)

            if isinstance(self.train_dataset, Subset):
                self.train_dataset.dataset._load_images = True
            else:
                self.train_dataset._load_images = True

            # close reading
            self.train_dataset.data._close()

            # generate random labels to remove
            removed_labels = torch.randperm(len(self.labels))[: self._dal_args.n_classes_removed]
            logger.info(f"Removed labels = {removed_labels}")

            labels_inv_mask = None
            for r_label in removed_labels:
                if labels_inv_mask is None:
                    labels_inv_mask = labels_per_sample[train_indices] == r_label
                else:
                    labels_inv_mask = labels_inv_mask | (labels_per_sample[train_indices] == r_label)
            labels_inv_mask = ~labels_inv_mask

            # get train indices where N randomly selected labels are not present
            self.update_dataset_labels(indices_to_add=train_indices[labels_inv_mask][:labeled_dataset_size])
        else:
            self.update_dataset_labels(indices_to_add=train_indices[:labeled_dataset_size])

    def update_dataset_labels(self, indices_to_add=None):
        import torch
        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

        logger = getLogger(DEFAULT_LOGGER_NAME)

        if indices_to_add is not None:
            self._labeled_indices_mask[indices_to_add] = True

        # get indices of labeled and unlabeled instances
        self._labeled_indices = torch.arange(0, self._pool_size)[self._labeled_indices_mask]
        self._unlabeled_indices = torch.arange(0, self._pool_size)[~self._labeled_indices_mask]

        # get number of labels for debugging
        # labels = {}
        # for index in self._labeled_indices.tolist():
        #     if self.train_dataset[index]["label"] not in labels:
        #         labels[self.train_dataset[index]["label"]] = 0
        #     labels[self.train_dataset[index]["label"]] += 1
        # logger.info(f"Number samples per label: {labels}")
        # exit()

        if len(self.train_dataset.indices) > 0:
            self._labeled_dataset.indices = torch.tensor(self.train_dataset.indices)[self._labeled_indices].tolist()
            self._unlabeled_dataset.indices = torch.tensor(self.train_dataset.indices)[self._unlabeled_indices].tolist()
        else:
            self._labeled_dataset.indices = self._labeled_indices.tolist()
            self._unlabeled_dataset.indices = self._unlabeled_indices.tolist()

        logger.info("Updated dataset labels.")

    def print_label_summary(self):
        from xai_torch.utilities.logging_utils import DEFAULT_LOGGER_NAME

        logger = getLogger(DEFAULT_LOGGER_NAME)
        logger.info(f"labeled set size = {len(self._labeled_dataset)}")
        logger.info(f"Unlabeled set size = {len(self._unlabeled_dataset)}")
        if self._pseudo_labeled_dataset is not None:
            logger.info(f"Pseudo labeled set size = {len(self._pseudo_labeled_dataset)}")

    def load_state(self, state):
        import copy

        self._labeled_indices_mask = state["labeled_indices_mask"]
        self._labeled_indices = state["labeled_indices"]
        self._unlabeled_indices = state["unlabeled_indices"]
        self._labeled_dataset.indices = self._labeled_indices.tolist()
        self._unlabeled_dataset.indices = self._unlabeled_indices.tolist()
        if "pseudo_label_indices" in state and "pseudo_labels" in state:
            self._pseudo_label_indices = state["pseudo_label_indices"]
            self._pseudo_labels = state["pseudo_labels"]
            self._pseudo_labeled_dataset = PseudoLabelsDataset(
                copy.deepcopy(self._labeled_dataset), state["pseudo_label_indices"], state["pseudo_labels"]
            )

    def val_dataloader(self, subset_size=None) -> DataLoader:
        """
        Defines the torch dataloader for validation dataset.
        """

        import ignite.distributed as idist
        from torch.utils.data import SequentialSampler

        if self._dal_args.val_samples_ratio is not None:
            val_subset_size = int(len(self._labeled_dataset) * self._dal_args.val_samples_ratio)
            val_subset = Subset(
                self.val_dataset,
                range(0, val_subset_size),
            )
        else:
            val_subset = self.val_dataset

        data_loader_args = self._args.data_args.data_loader_args
        if idist.get_world_size() > 1:
            if len(val_subset) % idist.get_world_size() != 0:
                self._logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )

            # let ignite handle distributed sampler
            sampler = None
        else:
            sampler = SequentialSampler(val_subset)

        return idist.auto_dataloader(
            val_subset,
            sampler=sampler,
            batch_size=data_loader_args.per_device_eval_batch_size,
            collate_fn=self._collate_fns.val,
            num_workers=data_loader_args.dataloader_num_workers,
            pin_memory=data_loader_args.pin_memory,
            drop_last=False,  # drop last is always false for validation
        )

    def setup_train_dataloader(
        self, dataset: Dataset, batch_size: Optional[int] = None, collate_fn: Optional[Callable] = None
    ):
        """
        Defines the torch dataloader for train dataset.
        """

        import ignite.distributed as idist
        import torch
        from torch.utils.data import RandomSampler, SequentialSampler
        from xai_torch.core.data.data_samplers.factory import BatchSamplerFactory

        # setup sampler
        data_loader_args = self._args.data_args.data_loader_args
        if data_loader_args.shuffle_data:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        # setup custom batch sampler
        return idist.auto_dataloader(
            dataset,
            sampler=sampler,
            batch_size=data_loader_args.per_device_train_batch_size if batch_size is None else batch_size,
            collate_fn=self._collate_fns.train if collate_fn is None else collate_fn,
            num_workers=data_loader_args.dataloader_num_workers,
            pin_memory=data_loader_args.pin_memory,
            drop_last=True if idist.get_world_size() > 1 else data_loader_args.dataloader_drop_last,
        )

    def train_dataloader(self):
        return self.labeled_dataloader()

    def labeled_dataloader(self):
        return self.setup_train_dataloader(self._labeled_dataset)

    def unlabeled_dataloader(self, batch_size: Optional[int] = None, unlabeled_subset_size: int = None):
        if unlabeled_subset_size is not None:
            import copy

            import torch

            unlabeled_subset = copy.deepcopy(self._unlabeled_dataset)
            r = torch.randperm(len(unlabeled_subset.indices))
            unlabeled_subset.indices = torch.tensor(unlabeled_subset.indices)[r][:unlabeled_subset_size].tolist()
            return self.setup_test_dataloader(unlabeled_subset, batch_size=batch_size), unlabeled_subset
        else:
            return self.setup_test_dataloader(self._unlabeled_dataset, batch_size=batch_size)

    def get_joint_dataset_loader(self, collate_fn: Optional[Callable] = None):
        return self.setup_train_dataloader(
            JointDataset(self._labeled_dataset, self._unlabeled_dataset), collate_fn=collate_fn
        )

    def prepare_pseudo_labels_dataset(self, pseudo_label_indices, pseudo_labels):
        import copy

        # copy the labeled dataset and add pseudo label samples in it
        self._pseudo_label_indices = pseudo_label_indices.tolist()
        self._pseudo_labels = pseudo_labels
        self._pseudo_labeled_dataset = PseudoLabelsDataset(
            copy.deepcopy(self._labeled_dataset), self._pseudo_label_indices, self._pseudo_labels
        )

    def pseudo_labeled_dataloader(self):
        return self.setup_train_dataloader(self._pseudo_labeled_dataset)


class PseudoLabelsDataset(Dataset):
    def __init__(self, wrapped_dataset: DatasetBase, pseudo_label_indices, pseudo_labels) -> None:
        super().__init__()
        self._wrapped_dataset = wrapped_dataset

        self._wrapped_dataset.indices += pseudo_label_indices
        self._pseudo_labels_map = {}
        for idx, label in zip(pseudo_label_indices, pseudo_labels):
            self._pseudo_labels_map[idx] = label

    def __getitem__(self, idx):
        sample = self._wrapped_dataset[idx]

        unwrapped_idx = self._wrapped_dataset.indices[idx]
        if unwrapped_idx in self._pseudo_labels_map.keys():
            sample[DataKeys.LABEL] = self._pseudo_labels_map[unwrapped_idx]
        return sample

    def __len__(self):
        return len(self._wrapped_dataset)
