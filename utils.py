import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F

from typing import Optional
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter
from torch_geometric.transforms import GDC
from torch.distributions import Uniform, Beta
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, subgraph
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.utils import to_dense_adj, dense_to_sparse,sort_edge_index

from typing import *
import os
import torch
import random
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams



def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


'''def build_dgl_graph(edge_index: torch.Tensor) -> dgl.DGLGraph:
    row, col = edge_index
    return dgl.graph((row, col))'''


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


###################
def get_adj_tensor(edge_index):
    # from edge index to scipy coo sparse matrix
    coo = to_scipy_sparse_matrix(edge_index)
    # from scipy coo sparse matrix to csr fromat
    adj = coo.tocsr()
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph should be unweighted"
    
    adj = torch.LongTensor(adj.todense())
    
    return adj
    
    
def get_normalize_adj_tensor(adj, device='cuda:0'):
    device = torch.device(device if adj.is_cuda else "cpu")
    
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    
    return mx


def get_spectral_weights(data):
    edge_index_ = to_undirected(data.edge_index)
    edge_index, edge_weight = get_laplacian(edge_index_, None, 'sym')
    return edge_index, edge_weight

##################

def permute(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute node embeddings or features.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Embeddings or features resulting from permutation.
    """
    return x[torch.randperm(x.size(0))]


def get_mixup_idx(x: torch.Tensor) -> torch.Tensor:
    """
    Generate node IDs randomly for mixup; avoid mixup the same node.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Random node IDs.
    """
    mixup_idx = torch.randint(x.size(0) - 1, [x.size(0)])
    mixup_self_mask = mixup_idx - torch.arange(x.size(0))
    mixup_self_mask = (mixup_self_mask == 0)
    mixup_idx += torch.ones(x.size(0), dtype=torch.int) * mixup_self_mask
    return mixup_idx


def mixup(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Randomly mixup node embeddings or features with other nodes'.

    Args:
        x: The latent embedding or node feature.
        alpha: The hyperparameter controlling the mixup coefficient.

    Returns:
        torch.Tensor: Embeddings or features resulting from mixup.
    """
    device = x.device
    mixup_idx = get_mixup_idx(x).to(device)
    lambda_ = Uniform(alpha, 1.).sample([1]).to(device)
    x = (1 - lambda_) * x + lambda_ * x[mixup_idx]
    return x


def multiinstance_mixup(x1: torch.Tensor, x2: torch.Tensor,
                        alpha: float, shuffle=False) -> (torch.Tensor, torch.Tensor):
    """
    Randomly mixup node embeddings or features with nodes from other views.

    Args:
        x1: The latent embedding or node feature from one view.
        x2: The latent embedding or node feature from the other view.
        alpha: The mixup coefficient `\lambda` follows `Beta(\alpha, \alpha)`.
        shuffle: Whether to use fixed negative samples.

    Returns:
        (torch.Tensor, torch.Tensor): Spurious positive samples and the mixup coefficient.
    """
    device = x1.device
    lambda_ = Beta(alpha, alpha).sample([1]).to(device)
    if shuffle:
        mixup_idx = get_mixup_idx(x1).to(device)
    else:
        mixup_idx = x1.size(0) - torch.arange(x1.size(0)) - 1
    x_spurious = (1 - lambda_) * x1 + lambda_ * x2[mixup_idx]

    return x_spurious, lambda_


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def dropout_feature(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
    return F.dropout(x, p=1. - drop_prob)


class AugmentTopologyAttributes(object):
    def __init__(self, pe=0.5, pf=0.5):
        self.pe = pe
        self.pf = pf

    def __call__(self, x, edge_index):
        edge_index = dropout_adj(edge_index, p=self.pe)[0]
        x = drop_feature(x, self.pf)
        return x, edge_index


def get_feature_weights(x, centrality, sparse=True):
    if sparse:
        x = x.to(torch.bool).to(torch.float32)
    else:
        x = x.abs()
    w = x.t() @ centrality
    w = w.log()

    return normalize(w)


def drop_feature_by_weight(x, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold, torch.ones_like(weights) * threshold)  # clip
    drop_mask = torch.bernoulli(weights).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def get_eigenvector_weights(data):
    def _eigenvector_centrality(data):
        graph = to_networkx(data)
        x = nx.eigenvector_centrality_numpy(graph)
        x = [x[i] for i in range(data.num_nodes)]
        return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

    evc = _eigenvector_centrality(data)
    scaled_evc = evc.where(evc > 0, torch.zeros_like(evc))
    scaled_evc = scaled_evc + 1e-8
    s = scaled_evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]

    return normalize(s_col), evc


def get_degree_weights(data):
    edge_index_ = to_undirected(data.edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[data.edge_index[1]].to(torch.float32)
    scaled_deg_col = torch.log(deg_col)

    return normalize(scaled_deg_col), deg


def get_pagerank_weights(data, aggr: str = 'sink', k: int = 10):
    def _compute_pagerank(edge_index, damp: float = 0.85, k: int = 10):
        num_nodes = edge_index.max().item() + 1
        deg_out = degree(edge_index[0])
        x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

        for i in range(k):
            edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
            agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

            x = (1 - damp) * x + damp * agg_msg

        return x

    pv = _compute_pagerank(data.edge_index, k=k)
    pv_row = pv[data.edge_index[0]].to(torch.float32)
    pv_col = pv[data.edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

    return normalize(s), pv


def drop_edge_by_weight(edge_index, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold, torch.ones_like(weights) * threshold)
    drop_mask = torch.bernoulli(1. - weights).to(torch.bool)

    return edge_index[:, drop_mask]


class AdaptivelyAugmentTopologyAttributes(object):
    def __init__(self, edge_weights, feature_weights, pe=0.5, pf=0.5, threshold=0.7):
        self.edge_weights = edge_weights
        self.feature_weights = feature_weights
        self.pe = pe
        self.pf = pf
        self.threshold = threshold

    def __call__(self, x, edge_index):
        edge_index = drop_edge_by_weight(edge_index, self.edge_weights, self.pe, self.threshold)
        x = drop_feature_by_weight(x, self.feature_weights, self.pf, self.threshold)

        return x, edge_index


def get_subgraph(x, edge_index, idx):
    adj = to_scipy_sparse_matrix(edge_index).tocsr()
    x_sampled = x[idx]
    edge_index_sampled = from_scipy_sparse_matrix(adj[idx, :][:, idx])
    return x_sampled, edge_index_sampled


def sample_nodes(x, edge_index, sample_size):
    idx = torch.randperm(x.size(0))[:sample_size]
    return get_subgraph(x, edge_index, idx), idx


def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight


def get_sparse_adj(edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                   add_self_loop: bool = True) -> torch.sparse.Tensor:
    num_nodes = edge_index.max().item() + 1
    num_edges = edge_index.size(1)

    if edge_weight is None:
        edge_weight = torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)

    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, num_nodes, normalization='sym')

    adj_t = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    return adj_t.t()


def compute_markov_diffusion(
        edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
        alpha: float = 0.1, degree: int = 10,
        sp_eps: float = 1e-3, add_self_loop: bool = True):
    adj = get_sparse_adj(edge_index, edge_weight, add_self_loop)

    z = adj.to_dense()
    t = adj.to_dense()
    for _ in range(degree):
        t = (1.0 - alpha) * torch.spmm(adj, t)
        z += t
    z /= degree
    z = z + alpha * adj

    adj_t = z.t()

    return GDC().sparsify_dense(adj_t, method='threshold', eps=sp_eps)


def coalesce_edge_index(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.FloatTensor):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    edge_weights = edge_weights if edge_weights is not None else torch.ones((num_edges,), dtype=torch.float32, device=edge_index.device)

    return coalesce(edge_index, edge_weights, m=num_nodes, n=num_nodes)


def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1    
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)


    return coalesce_edge_index(edge_index)[0]


def switch_edge(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, ratio1: float = 0.5, ratio2: float = 0.5) -> (torch.Tensor, Optional[torch.Tensor]):
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1    
    num_add = int(num_edges * ratio1)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    
    edge_index, edge_weight = dropout_adj(edge_index, edge_attr=edge_weight, p=ratio2)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)


    return coalesce_edge_index(edge_index, edge_weight)[0]


def drop_node(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, keep_prob: float = 0.5) -> (torch.Tensor, Optional[torch.Tensor]):
    num_nodes = edge_index.max().item() + 1
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = Bernoulli(probs)

    subset = dist.sample().to(torch.bool).to(edge_index.device)
    edge_index, edge_weight = subgraph(subset, edge_index, edge_weight)

    return edge_index, edge_weight


def random_walk_subgraph(edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None, batch_size: int = 1000, length: int = 10):
    num_nodes = edge_index.max().item() + 1

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    start = torch.randint(0, num_nodes, size=(batch_size, ), dtype=torch.long).to(edge_index.device)
    node_idx = adj.random_walk(start.flatten(), length).view(-1)

    edge_index, edge_weight = subgraph(node_idx, edge_index, edge_weight)

    return edge_index, edge_weight

def visualize_a_graph(i,edge_index, edge_att, node_label, dataset_name, tag,coor=None, norm=False, mol_type=None, nodesize=300):
    plt.clf()
    if norm:
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')
    #plt.show()
    #plt.rcParams['figure.figsize']=(20, 7.2)
    plt.savefig('./Figure/{}_'.format(i) + tag +'.svg',dpi=300)
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image
    
def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]