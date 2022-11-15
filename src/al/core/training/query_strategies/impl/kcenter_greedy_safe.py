from __future__ import annotations, division, print_function

import numpy as np
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked


@register_strategy("kcenter_greedy_safe")
class KCenterSamplingSafe(QueryStrategy):
    def get_embeddings_from_train_data(self):
        import torch

        embeddings = []
        self.attach_embedding_hook(embeddings)
        self._prediction_engine.run(self._datamodule.setup_test_dataloader(self._datamodule.train_dataset))
        self.detach_embedding_hook()

        embeddings = torch.cat(embeddings)
        return embeddings

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.all_pts[centers]  # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric="euclidean")

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def update_dist_chunked(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:

            def min_dist(chunk, start):
                return np.min(chunk, axis=1).reshape(-1, 1)

            x = self.all_pts[centers]  # pick only centers
            pw_dists = pairwise_distances_chunked(
                self.all_pts, x, metric="euclidean", working_memory=20000, reduce_func=min_dist
            )
            min_dists = []
            for d in pw_dists:
                min_dists.append(d)
            min_dists = np.concatenate(min_dists)

            if self.min_distances is None:
                self.min_distances = min_dists
            else:
                self.min_distances = np.minimum(self.min_distances, min_dists)

    def query(self, n_samples: int):
        import copy

        import numpy as np
        import tqdm
        from sklearn.decomposition import PCA

        embeddings = self.get_embeddings_from_train_data()
        embeddings = embeddings.numpy()

        # downsampling embeddings if feature dim > 50 to make it faster
        print("Reducing embedding dimensions...")
        if len(embeddings[0]) > 64:
            pca = PCA(n_components=64)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)

        self.all_pts = embeddings
        self.dset_size = len(self.all_pts)
        self.min_distances = None
        self.already_selected = []

        # initially updating the distances
        already_selected = copy.deepcopy(self._datamodule.labeled_indices)
        print("Computing pairwise distances...")
        self.update_dist_chunked(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        new_batch = []
        for _ in tqdm.tqdm(range(n_samples)):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)

            assert ind not in already_selected
            self.update_dist_chunked([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)

        max_distance = max(self.min_distances)
        return new_batch
