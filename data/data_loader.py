import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from utils import to_onehot, normalize_adjacent


def load_data(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = to_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # normalize & convert to torch.Tensor
    features = torch.tensor(np.array(features.todense()), dtype=torch.float32)
    features = F.normalize(features, p=1, dim=1)
    adj = normalize_adjacent(adj + sp.eye(adj.shape[0]))
    adj = torch.tensor(np.array(adj.todense()), dtype=torch.float32)
    labels = torch.tensor(np.where(labels)[1], dtype=torch.int64)

    # train/val/test splitting
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.tensor(idx_train, dtype=torch.int64)
    idx_val = torch.tensor(idx_val, dtype=torch.int64)
    idx_test = torch.tensor(idx_test, dtype=torch.int64)

    return adj, features, labels, idx_train, idx_val, idx_test