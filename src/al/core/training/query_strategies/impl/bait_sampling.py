from builtins import print
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
from torch.nn import functional as F
import numpy as np

# from sklearn.externals.six import string_types

import numpy as np
import gc
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch.autograd import Variable
import pdb
from torch.nn import functional as F
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


def select(X, K, fisher, iterates, lamb=1, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection, over-sample by 2x
    print("forward selection...", flush=True)
    over_sample = 2
    for i in range(int(over_sample * K)):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = (
            torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo("float32").max
        )
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1
        ).sum(-1)

        # clear out gpu memory
        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        print(i, ind, traceEst[ind], flush=True)

        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    print("backward pruning...", flush=True)
    for i in range(len(indsAll) - K):

        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1
        ).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
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


@register_strategy("bait_sampling")
class BaitSampling(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        lamb: int = 1,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._lamb = lamb

    def get_exp_grad_embedding(self):
        from ignite.utils import convert_tensor

        embedding_model = ModelWrapper(self._model._torch_model.model)

        emb_dim = 128
        embedding_model.eval()
        num_classes = self._datamodule.num_labels
        total_samples = len(self._datamodule.train_dataset)
        embedding = np.zeros([total_samples, num_classes, emb_dim * num_classes])
        for ind in range(num_classes):
            unlabeled_dataloader = self._datamodule.setup_test_dataloader(self._datamodule.train_dataset)
            with torch.no_grad():
                start_idx = 0
                for batch in unlabeled_dataloader:
                    batch = convert_tensor(batch, device=self._device)
                    idxs = range(start_idx, start_idx + len(batch[DataKeys.LABEL]))
                    cout, out = embedding_model(**batch)
                    out = out.data.cpu().numpy()
                    print("Reducing embedding dimensions...")
                    if len(out[0]) > 128:
                        pca = PCA(n_components=128)
                        out = pca.fit_transform(out)
                    out = out.astype(np.float16)

                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    for j in range(len(batch[DataKeys.LABEL])):
                        for c in range(num_classes):
                            if c == ind:
                                embedding[idxs[j]][ind][emb_dim * c : emb_dim * (c + 1)] = deepcopy(out[j]) * (
                                    1 - batchProbs[j][c]
                                )
                            else:
                                embedding[idxs[j]][ind][emb_dim * c : emb_dim * (c + 1)] = deepcopy(out[j]) * (
                                    -1 * batchProbs[j][c]
                                )
                        embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
                    start_idx = len(batch[DataKeys.LABEL])
        return torch.Tensor(embedding)

    def query(self, n_samples):
        import tqdm

        labeled_mask = self._datamodule.labeled_indices_mask
        # get low-rank point-wise fishers
        xt = self.get_exp_grad_embedding()

        # get fisher
        print("getting fisher matrix...", flush=True)
        batchSize = 100  # should be as large as gpu memory allows
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        for i in tqdm.tqdm(range(int(np.ceil(len(self._datamodule.train_dataset) / batchSize)))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        print("xt", xt.shape)
        print("labeled_mask", labeled_mask.shape)
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[labeled_mask]
        for i in tqdm.tqdm(range(int(np.ceil(len(xt2) / batchSize)))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = select(
            xt[self._datamodule.unlabeled_indices],
            n_samples,
            fisher,
            init,
            lamb=self._lamb,
            nLabeled=np.sum(labeled_mask.cpu().numpy()),
        )
        return self._datamodule.unlabeled_indices[chosen]
