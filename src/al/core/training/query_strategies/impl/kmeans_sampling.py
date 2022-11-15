from __future__ import annotations

import torch
from al.core.data.active_learning_datamodule import ActiveLearningDataModule
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from xai_torch.core.models.xai_model import XAIModel


@register_strategy("kmeans_sampling")
class KMeansSampling(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        unlabeled_subset_size: int = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._unlabeled_subset_size = unlabeled_subset_size

    def query(self, n_samples: int):
        import numpy as np
        from sklearn.decomposition import PCA

        embeddings, unlabeled_subset = self.get_embeddings_from_unlabeled_data()
        # unlabeled_subset  = None
        # embeddings = torch.randn((n_samples, 128))

        # downsampling embeddings if feature dim > 50 to make it faster
        print("Reducing embedding dimensions...")
        if len(embeddings[0]) > 128:
            pca = PCA(n_components=128)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)

        cluster_learner = KMeans(n_clusters=n_samples, verbose=1)
        cluster_learner.fit(embeddings)

        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers) ** 2
        dis = dis.sum(axis=1)

        q_idxs = np.array(
            [
                np.arange(embeddings.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()]
                for i in range(n_samples)
            ]
        )

        if unlabeled_subset is not None:
            return torch.tensor(unlabeled_subset.indices)[q_idxs]
        else:
            return torch.tensor(self._datamodule.unlabeled_indices)[q_idxs]

    def get_embeddings_from_unlabeled_data(self, return_logits=False):
        import torch

        unlabeled_subset = None
        if self._unlabeled_subset_size is not None:
            dataloader, unlabeled_subset = self._datamodule.unlabeled_dataloader(
                unlabeled_subset_size=self._unlabeled_subset_size
            )
        else:
            dataloader = self._datamodule.unlabeled_dataloader()

        embeddings = []
        self.attach_embedding_hook(embeddings)
        results = self._prediction_engine.run(dataloader)
        self.detach_embedding_hook()

        embeddings = torch.cat(embeddings)
        if not return_logits:
            return embeddings, unlabeled_subset
        return embeddings, results.metrics["logits"].cpu(), unlabeled_subset
