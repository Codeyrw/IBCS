import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import random
import os
import os.path as osp
import sys
sys.path.append('../')
import copy
import torch
from torch import nn
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,global_add_pool,global_mean_pool
from conv_layers import GINConv, GINEConv
from torch_geometric.nn.inits import uniform
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import subgraph, is_undirected
from sklearn.metrics import f1_score,accuracy_score
from utils import seed_everything,visualize_a_graph
from get_data_loaders import get_data_loaders
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from torch_geometric.nn import InstanceNorm
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' 
    torch.use_deterministic_algorithms(True)
###################### Backbone Model ######################
class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
class GConv(nn.Module):
    def __init__(self, gnn,input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        self.dropout_p=0.3
        if gnn == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn =='GIN':
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GINConv(
                    nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    ), train_eps=False))
                else:
                    self.layers.append(GINConv(
                    nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    ), train_eps=False))
        elif gnn =='GAT':
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GATConv(input_dim, hidden_dim,heads=2))
                else:
                    self.layers.append(GATConv(2*hidden_dim, hidden_dim))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(SAGEConv(input_dim, hidden_dim))
                else:
                    self.layers.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch,edge_attr=None,edge_atten=None):
        z=x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index,edge_attr=edge_attr, edge_atten=edge_atten)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout_p, training=self.training)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, gs[-1]

def calc_loss(x, x_aug,temperature=0.2):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    
    loss_0 = pos_sim / (sim_matrix.sum(dim=0) )
    loss_1 = pos_sim / (sim_matrix.sum(dim=1) )

    loss_0 = - torch.log(loss_0).mean()
    loss_1 = - torch.log(loss_1).mean()
    loss_re = (loss_0 + loss_1) / 2.0
    return -loss_re
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

class FC_sub(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(FC_sub, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            InstanceNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2*hidden_dim, hidden_dim),
            InstanceNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def visualize_results(test_set, idx, epoch, tag, use_edge_attr,dataset,encoder_model,multi_label,device):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        _, _, _, _,_, batch_att= val_vis(dataset,encoder_model,data.to(device),multi_label,device)

        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if dataset == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif dataset == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif dataset == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif dataset == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index.to(device), edge_attr=batch_att.to(device))

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(i,viz_set[i].edge_index, edge_att, node_label, dataset,tag, coor=coor,norm=True, mol_type=mol_type)

class Encoder(torch.nn.Module):
    def __init__(self,args,n_class,gcn1,gcn_con,mlp1,mlp2,mlp3,context,objects,edge_mlp):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn_con = gcn_con
        self.mlp1 = mlp1
        self.mlp_IB = mlp2
        self.mlp3 = mlp3
        self.context = context
        self.objects = objects
        self.edge_mlp = edge_mlp
        self.n_class = n_class
        self.mlp_co = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=self.n_class)
        self.mlp_o = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=self.n_class)
        self.mlp_c = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=self.n_class)
        self.mlp_co_s = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=self.n_class)
        self.optimizer_co = Adam(self.mlp_co.parameters(),lr=args.lr)
        self.optimizer_o = Adam(self.mlp_o.parameters(),lr=args.lr)
        self.optimizer_c = Adam(self.mlp_c.parameters(),lr=args.lr)
        self.optimizer_co_s = Adam(self.mlp_co_s.parameters(),lr=0.001)
    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    def sampling(self, att_log_logits,training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att
    def forward(self, data,flag=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        z1, g1 = self.gcn1(x, edge_index, batch)

        IB_score = self.mlp_IB(z1)
        assignment = F.softmax(IB_score,dim=1)
        assignment = F.gumbel_softmax(assignment, tau = 1, dim = -1)

        edge_weight_o = lift_node_att_to_edge_att(assignment[:,0], data.edge_index).unsqueeze(1)
        edge_weight_c = lift_node_att_to_edge_att(assignment[:,1], data.edge_index).unsqueeze(1)

        z_M,_ = self.context(x, edge_index, batch,edge_attr=None,edge_atten=edge_weight_o)
        z_res,_ = self.objects(x,edge_index, batch,edge_attr=None,edge_atten=edge_weight_c)
        
        g_M = global_add_pool(z_M, batch)
        g_res = global_add_pool(z_res, batch)
        
        num = g_res.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        g_co = g_res[random_idx] + g_M 
        g_co_s = g_res + g_M[random_idx]

        y_shuf= data.y.squeeze(1).to(torch.long)[random_idx]
        
        h_co =self.mlp_co(g_co)
        h_M = self.mlp_o(g_M)
        h1 = self.mlp1(g1)
        h_res = self.mlp_c(g_res)
        h_co_s = self.mlp_co_s(g_co_s)
        

        proj_M = self.mlp3(g_M.detach())
        proj1 = self.mlp3(g1.detach()) 
        return h1,g1,h_M, g_M,proj1,proj_M,assignment,h_co,h_res,edge_weight_o,h_co_s,y_shuf


        
def train_M(encoder_model, data, optimizer_gcn_con,optimizer_mlp3,cur, device):
    encoder_model.train()
    epoch_loss = 0
    CE = nn.CrossEntropyLoss()
    optimizer_mlp3.zero_grad()
    optimizer_gcn_con.zero_grad()
    data = data.to(device)
    if data.x.shape[1]==0:
        num_nodes = data.batch.size(0)
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

    h1,g1,h_M,g_M,proj1,proj_M,ass,h_co,h_res,_,_,_ = encoder_model(data,True)

    #----------------------------------------------------#
    con_loss = -1*calc_loss(proj1,proj_M)


    loss = con_loss 
    '''print('loss_train2: {:.4f}'.format(cls_loss2.item()),
          'con_loss: {:.4f}'.format(con_loss.item())) '''
    loss.backward()
    optimizer_mlp3.step()
    optimizer_gcn_con.step()
    
    return loss

def train(encoder_model, data, optimizer_mlp1,optimizer_gcn1,optimizer_mlp2, optimizer_context,optimizer_objects,optimizer_edge,device):
    encoder_model.train()
    epoch_loss = 0
    CE = nn.CrossEntropyLoss()
    
    data = data.to(device)
    optimizer_mlp1.zero_grad()
    optimizer_gcn1.zero_grad()
    optimizer_mlp2.zero_grad()
    optimizer_context.zero_grad()
    optimizer_objects.zero_grad()
    optimizer_edge.zero_grad()
    encoder_model.optimizer_o.zero_grad()
    encoder_model.optimizer_co.zero_grad()
    encoder_model.optimizer_c.zero_grad()
    encoder_model.optimizer_co_s.zero_grad()
    if data.x.shape[1]==0:
        num_nodes = data.batch.size(0)
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
    
    h1,g1,h_M,g_M,proj1,proj_M,ass,h_co,h_res,_ ,h_co_s,y_shuf= encoder_model(data,True)
    ce = CE(ass,ass)
    con_loss = calc_loss(proj1,proj_M)
    cls_loss0 = CE(h1,data.y.squeeze(1).to(torch.long))
    cls_loss1 = CE(h_M,data.y.squeeze(1).to(torch.long))
    cls_loss4= CE(h_co,data.y.squeeze(1).to(torch.long))
    uniform_target = torch.ones_like(h_res, dtype=torch.float).to(device) / h_res.shape[1]

    cls_loss2 = CE(h_res,data.y.squeeze(1).to(torch.long))
    cls_loss3 = CE(h_co_s,y_shuf)
    v=torch.ones(data.edge_index.shape[1]).cuda()
    adj = torch.sparse_coo_tensor(data.edge_index, v, (data.batch.size(0), data.batch.size(0))).to_dense().cuda()
    lian = torch.matmul(torch.matmul(ass.T,adj),ass)
    norm_lian = F.normalize(lian, p=1, dim=1)

    coc_loss = torch.norm(norm_lian-torch.eye(2).cuda(),p='fro')/(torch.max(data.batch)+1)

    
    loss = 1*cls_loss0 + 1*cls_loss1 + 0.1*con_loss   + coc_loss + 2*cls_loss2 + 3*cls_loss3 
    loss.backward()
    optimizer_mlp1.step()
    optimizer_gcn1.step()
    optimizer_mlp2.step()
    optimizer_context.step()
    optimizer_objects.step()
    optimizer_edge.step()
    encoder_model.optimizer_o.step()
    encoder_model.optimizer_co.step()
    encoder_model.optimizer_c.step()
    encoder_model.optimizer_co_s.step()
    return loss
def get_precision_at_k(att, exp_labels, k, batch, edge_index):
    precision_at_k = []
    for i in range(batch.max()+1):
        nodes_for_graph_i = batch == i
        edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
        labels_for_graph_i = exp_labels[edges_for_graph_i.cpu()]
        mask_log_logits_for_graph_i = att[edges_for_graph_i.cpu()]
        precision_at_k.append(labels_for_graph_i.cpu()[np.argsort(-mask_log_logits_for_graph_i.cpu())[:k]].sum().item() / k)
    return precision_at_k

def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data
    
def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att
        
def test(dataset_name,encoder_model, dataloader,multi_label,device):
    encoder_model.eval()
    CE = nn.CrossEntropyLoss()
    k=5
    atts = []
    h_Ms = []
    all_exp_labels=[]
    all_pre=[]
    y=[]
    for data in dataloader:
        data = process_data(data, False)
        data =data.to(device)
        h1, g1,h_M,g_M,_,_,ass,_,_,att,_,_ = encoder_model(data,False)
        att = lift_node_att_to_edge_att(ass[:,0], data.edge_index)
        exp_labels = data.edge_label.data.cpu()

        precision_at_k = get_precision_at_k(att.data.cpu(), exp_labels.data.cpu(), k, data.batch, data.edge_index.data.cpu())
        h_Ms.append(h_M)
        atts.append(att)
        y.append(data.y.squeeze(1).data.cpu())
        all_exp_labels.append(exp_labels)
        all_pre.append(precision_at_k)
    exp_labels, attss = torch.cat(all_exp_labels), torch.cat(atts),
    clf_labels, clf_logits = torch.cat(y), torch.cat(h_Ms)
    
    att_auroc, precision, clf_acc, clf_roc = get_eval_score(dataset_name,exp_labels, attss, precision_at_k, clf_labels, clf_logits, False,multi_label)
    return att_auroc, precision, clf_acc, clf_roc


def val(dataset_name,encoder_model, dataloader, multi_label, device):
    encoder_model.eval()
    CE = nn.CrossEntropyLoss()
    k=5
    atts = []
    h_Ms = []
    all_exp_labels=[]
    all_pre=[]
    loss=0
    y=[]
    for data in dataloader:
        data = process_data(data, False)
        data =data.to(device)
        h1, g1,h_M,g_M,_,_,ass,_,_,att,_,_= encoder_model(data,False)
        att = lift_node_att_to_edge_att(ass[:,0], data.edge_index)
        exp_labels = data.edge_label.data.cpu()

        cls_loss = CE(h_M,data.y.squeeze(1).to(torch.long)).item()
        precision_at_k = get_precision_at_k(att.data.cpu(), exp_labels.data.cpu(), k, data.batch, data.edge_index.data.cpu())
        loss+=cls_loss
        h_Ms.append(h_M)
        atts.append(att)
        y.append(data.y.squeeze(1).data.cpu())
        all_exp_labels.append(exp_labels)
        all_pre.append(precision_at_k)
    exp_labels, attss = torch.cat(all_exp_labels), torch.cat(atts),
    clf_labels, clf_logits = torch.cat(y), torch.cat(h_Ms)
    
    att_auroc, precision, clf_acc, clf_roc = get_eval_score(dataset_name,exp_labels, attss, precision_at_k, clf_labels, clf_logits, False,multi_label)
    return att_auroc, precision, clf_acc, clf_roc, loss/len(dataloader.dataset),attss
def val_vis(dataset_name,encoder_model, data, multi_label, device):
    encoder_model.eval()
    CE = nn.CrossEntropyLoss()
    k=5
    
    data = process_data(data, False)
    data =data.to(device)
    h1, g1,h_M,g_M,_,_,ass,_,_,att,_,_= encoder_model(data,False)
    att = lift_node_att_to_edge_att(ass[:,0], data.edge_index)
    exp_labels = data.edge_label.data.cpu()

    cls_loss = CE(h_M,data.y.squeeze(1).to(torch.long)).item()
    precision_at_k = get_precision_at_k(att.data.cpu(), exp_labels.data.cpu(), k, data.batch, data.edge_index.data.cpu())
    loss+=cls_loss

    
    att_auroc, precision, clf_acc, clf_roc = get_eval_score(dataset_name,exp_labels, att, precision_at_k, data.y.squeeze(1).data.cpu(), h_M, False,multi_label)
    print(precision)
    return att_auroc, precision, clf_acc, clf_roc, loss,att
def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  
        preds = logits.argmax(dim=1).float()
    else:  
        preds = (logits.sigmoid() > 0.5).float()
    return preds
def get_eval_score(dataset_name,exp_labels, att, precision_at_k, clf_labels, clf_logits, batch,multi_label):
        clf_preds = get_preds(clf_logits, multi_label)
        clf_acc = 0 if multi_label else (clf_preds.cpu() == clf_labels.cpu()).sum().item() / clf_labels.shape[0]
        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in dataset_name:
            evaluator = Evaluator(name='-'.join(dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels.detach().cpu().numpy(), att.detach().cpu().numpy())
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]
     
        return  att_auroc, precision_at_k, clf_acc, clf_roc
def load_dataset(dataset_name="MUTAG"):
    graphs = TUDataset("./data/raw", dataset_name)
        
    if graphs.data.x is None:
        max_degree = 0
        degs = []
        for data in graphs:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 2000:
            graphs.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            graphs.transform = NormalizedDegree(mean, std)

    return graphs

def get_viz_idx(test_set, dataset_name,num_viz_samples):
    y_dist = test_set.data.y.numpy().reshape(-1)
    num_nodes = np.array([each.x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name == 'Graph-SST2':
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  
            candidate_set = np.nonzero(condi)[0]
        else:
            condi = (y_dist == each_class) 
            candidate_set = np.nonzero(condi)[0]

        idx = np.random.choice(candidate_set,num_viz_samples, replace=False)
        res.append((idx, tag))
    return res

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--viz_interval', type=int, default=5, help='visualize interval')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--dataset', type=str, default='Graph-SST2') 

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--outer_steps', type=int, default=1)

    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--temp_r', type=float, default=1e-2)
    #parser.add_argument('--aug_iter', type=int, default=20, help='iteration for augmentation')
    #parser.add_argument('--pf', type=float, default=0.4, help='feature masking probability')
    #parser.add_argument('--pe', type=float, default=0.2, help='edge perturbation probability')
    return parser.parse_args()
def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), [int(x) for x in dataset.data.y]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices
def main():
    args = arg_parse()
    setup_seed(0)
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    print(args.dataset)

    splits={'train':0.8,'valid':0.1,'test':0.1}
    mutag_x  = True if args.dataset=='MUTAG' else False
    data_dir= './data/'
    loaders, test_set, x_dim, num_class, aux_info = get_data_loaders(data_dir, args.dataset, args.batch_size, splits, 0,mutag_x)
    
    dataloader = loaders['train']  
    val_dataloader = loaders['valid']
    test_dataloader = loaders['test'] 
    
    gcn1 = GConv(gnn='GIN',input_dim=max(x_dim, 1), hidden_dim=args.hidden, num_layers=2).to(device)

    gcn_con = GConv(gnn='GIN',input_dim=max(x_dim, 1), hidden_dim=args.hidden, num_layers=2).to(device)
    mlp1 = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=num_class)
    mlp2 = FC_sub(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=2)
    mlp3 = FC(input_dim=args.hidden, hidden_dim=args.hidden, output_dim=args.hidden)

    context = GConv(gnn='GIN',input_dim=max(x_dim, 1), hidden_dim=args.hidden, num_layers=2).to(device)
    objects = GConv(gnn='GIN',input_dim=max(x_dim, 1), hidden_dim=args.hidden, num_layers=2).to(device)
    edge_att_mlp = FC_sub(input_dim=2*args.hidden, hidden_dim=args.hidden, output_dim=2)

    encoder_model = Encoder(args=args,n_class=num_class,gcn1=gcn1,gcn_con=gcn_con,mlp1=mlp1,mlp2=mlp2,mlp3=mlp3,context= context,objects=objects,edge_mlp=edge_att_mlp).to(device)

    
    optimizer_gcn1  = Adam(gcn1.parameters(),lr=args.lr)
    optimizer_mlp1 = Adam(mlp1.parameters(), lr=args.lr)
    optimizer_mlp2 = Adam(mlp2.parameters(), lr=args.lr)
    optimizer_mlp3 = Adam(mlp3.parameters(), lr=args.lr)
    optimizer_edge = Adam(edge_att_mlp.parameters(), lr=args.lr)
    optimizer_gcn_con = Adam(gcn_con.parameters(), lr=args.lr)
    optimizer_context = Adam(context.parameters(), lr=args.lr)
    optimizer_objects = Adam(objects.parameters(), lr=args.lr)

    best_acc=0
    best_loss=1e10
    best_val_epoch=0
    cnt=0
    num_viz_samples=64
    viz_set = get_viz_idx(test_set, args.dataset,num_viz_samples)
    for epoch in range(1, args.epoch+1):
        cur = np.log(1 + args.temp_r * epoch)
        cur = min(max(0.05, cur), 0.5)
        for data in dataloader:
            for i in range(int(args.inner_steps)):
                loss = train_M(encoder_model, data,optimizer_gcn_con, optimizer_mlp3,cur, device)
            for i in range(int(args.outer_steps)):
                loss = train(encoder_model, data, optimizer_mlp1,optimizer_gcn1,optimizer_mlp2,optimizer_context,optimizer_objects,optimizer_edge,device)
        
        att_auroc, precision, clf_acc, clf_roc, val_loss,_  = val(args.dataset,encoder_model, val_dataloader,aux_info['multi_label'],device)

        if clf_acc >best_acc or (clf_acc==best_acc and val_loss < best_loss):
            best_acc = clf_acc
            best_loss = val_loss
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder_model)
            cnt=0
        else:
            cnt+=1
        if cnt>=10:
            print('Early stopping')
            break
        with torch.no_grad():
            att_auroc, precision, clf_acc, clf_roc = test(args.dataset,best_model, test_dataloader,aux_info['multi_label'],device)
            print(f'(E): Test accuracy={clf_acc:.4f}, precision={precision:.4f}')
            '''if num_viz_samples != 0 and (epoch % args.viz_interval == 0 or epoch == args.epoch - 1):
                if aux_info['multi_label']:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    if tag=='class_0.0':
                      visualize_results(test_set, idx, epoch, tag, False,args.dataset,encoder_model,aux_info['multi_label'], device)'''
        print(epoch,best_acc)
    best_model.eval()
    with torch.no_grad():
        att_auroc, precision, clf_acc, clf_roc = test(args.dataset,best_model, test_dataloader,aux_info['multi_label'],device)

        print(f'(E): Test accuracy={clf_acc:.4f}, precision={precision:.4f}')

        
if __name__ == '__main__':
    main()
