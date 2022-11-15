from __future__ import annotations

import torch
from al.core.data.active_learning_datamodule import ActiveLearningDataModule
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from sklearn.cluster import KMeans
from xai_torch.core.models.xai_model import XAIModel
from sklearn.decomposition import PCA


@register_strategy("kcenter_greedy")
class KCenterSampling(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        labeled_subset_size: int = None,
        unlabeled_subset_size: int = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._labeled_subset_size = labeled_subset_size
        self._unlabeled_subset_size = unlabeled_subset_size

        if self._labeled_subset_size is not None:
            assert self._unlabeled_subset_size is not None

    def get_embeddings_from_train_data(self):
        import torch

        embeddings = []
        self.attach_embedding_hook(embeddings)
        self._prediction_engine.run(self._datamodule.setup_test_dataloader(self._datamodule.train_dataset))
        self.detach_embedding_hook()

        embeddings = torch.cat(embeddings)
        return embeddings

    def query(self, n_samples: int):
        if self._labeled_subset_size is not None:
            return self.subset_query(n_samples)
        else:
            return self.full_query(n_samples)

    def full_query(self, n_samples: int):
        import copy

        import numpy as np
        import tqdm

        labeled_indices_mask = copy.deepcopy(self._datamodule.labeled_indices_mask)
        embeddings = self.get_embeddings_from_train_data()
        embeddings = embeddings.numpy()

        #downsampling embeddings if feature dim > 50
        if len(embeddings[0]) > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)
        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_indices_mask), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_indices_mask, :][:, labeled_indices_mask]

        for _ in tqdm.tqdm(range(n_samples), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self._datamodule.pool_size)[~labeled_indices_mask][q_idx_]
            labeled_indices_mask[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_indices_mask, q_idx][:, None], axis=1)

        return np.arange(self._datamodule.pool_size)[(self._datamodule.labeled_indices_mask ^ labeled_indices_mask)]

    def subset_query(self, n_samples: int):
        import copy

        import numpy as np
        import tqdm

        labeled_indices_mask = copy.deepcopy(self._datamodule.labeled_indices_mask)
        embeddings = self.get_embeddings_from_train_data()
        embeddings = embeddings.numpy()

        if self._labeled_subset_size is not None:
            subset_labeled_indices = np.arange(0, embeddings.shape[0])[labeled_indices_mask][
                : self._labeled_subset_size
            ]
            subset_unlabeled_indices = np.arange(0, embeddings.shape[0])[~labeled_indices_mask][
                : self._labeled_subset_size
            ]

            # subset labelled embeddings
            labeled_embeddings = embeddings[subset_labeled_indices]
            unlabeled_embeddings = embeddings[subset_unlabeled_indices]

            # get labeled_indices_mask
            labeled_indices_mask = np.concatenate(
                [labeled_indices_mask[subset_labeled_indices], labeled_indices_mask[subset_unlabeled_indices]]
            )

            embeddings = np.concatenate([labeled_embeddings, unlabeled_embeddings])

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(embeddings.shape[0], 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_indices_mask, :][:, labeled_indices_mask]

        for _ in tqdm.tqdm(range(n_samples), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(labeled_indices_mask.shape[0])[~labeled_indices_mask][q_idx_]
            labeled_indices_mask[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_indices_mask, q_idx][:, None], axis=1)

        return subset_unlabeled_indices[labeled_indices_mask[len(subset_labeled_indices) :] == True]
