import torch.nn.functional as F
import numpy as np
import torch
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

import torch
from al.core.data.active_learning_datamodule import ActiveLearningDataModule
from xai_torch.core.models.xai_model import XAIModel
from copy import deepcopy

from xai_torch.core.constants import DataKeys
import numpy as np
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
import pdb
from torch.nn import functional as F
from scipy import stats
import numpy as np

# from sklearn.externals.six import string_types
from sklearn.metrics import pairwise_distances


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    print("#Samps\tTotal Distance")
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + "\t" + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._fc = model.fc
        self._model = torch.nn.Sequential(*list(model.children())[:-1])
        self.embedding_dim = self._fc.in_features

    def forward(self, image=None, label=None, stage=None):
        embeddings = self._model(image)
        return self._fc(embeddings.squeeze()), embeddings.squeeze()


@register_strategy("badge_sampling")
class BadgeSampling(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)

    def get_grad_embedding(self):
        from ignite.utils import convert_tensor

        embedding_model = ModelWrapper(self._model._torch_model.model)
        unlabeled_dataloader = self._datamodule.unlabeled_dataloader()
        emb_dim = embedding_model.embedding_dim
        embedding_model.eval()
        num_classes = self._datamodule.num_labels
        total_samples = len(self._datamodule._unlabeled_dataset)
        embedding = np.zeros([total_samples, emb_dim * num_classes])

        with torch.no_grad():
            start_idx = 0
            for batch in unlabeled_dataloader:
                batch = convert_tensor(batch, device=self._device)
                idxs = range(start_idx, start_idx + len(batch[DataKeys.LABEL]))
                cout, out = embedding_model(**batch)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(batch[DataKeys.LABEL])):
                    for c in range(num_classes):
                        # print("embedding", embedding.shape)
                        # print("idxs", idxs.shape)
                        if c == maxInds[j]:
                            embedding[idxs[j]][emb_dim * c : emb_dim * (c + 1)] = deepcopy(out[j]) * (
                                1 - batchProbs[j][c]
                            )
                        else:
                            embedding[idxs[j]][emb_dim * c : emb_dim * (c + 1)] = deepcopy(out[j]) * (
                                -1 * batchProbs[j][c]
                            )
                start_idx = len(batch[DataKeys.LABEL])
            return torch.Tensor(embedding)

    def query(self, n_samples):
        gradEmbedding = self.get_grad_embedding().numpy()
        chosen = init_centers(gradEmbedding, n_samples)
        return self._datamodule.unlabeled_indices[chosen[:n_samples]]
