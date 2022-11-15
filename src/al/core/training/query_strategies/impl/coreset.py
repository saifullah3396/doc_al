from __future__ import annotations, division, print_function

import numpy as np
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from sklearn.metrics import pairwise_distances


@register_strategy("coreset")
class Coreset(QueryStrategy):
    def get_embeddings_from_train_data(self):
        import torch

        embeddings = []
        self.attach_embedding_hook(embeddings)
        self._prediction_engine.run(self._datamodule.setup_test_dataloader(self._datamodule.train_dataset))
        self.detach_embedding_hook()

        embeddings = torch.cat(embeddings)
        return embeddings

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n_samples: int):
        import copy
        import numpy as np
        
        labeled_indices_mask = copy.deepcopy(self._datamodule.labeled_indices_mask)
        embeddings = self.get_embeddings_from_train_data()
        embeddings = embeddings.numpy()
        chosen = self.furthest_first(embeddings[~labeled_indices_mask, :], embeddings[labeled_indices_mask, :], n_samples)
        return np.sort(self._datamodule.unlabeled_indices[chosen])