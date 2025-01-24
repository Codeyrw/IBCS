import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from torch.nn import Linear, BatchNorm1d
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,global_add_pool,global_mean_pool,global_max_pool, GlobalAttention
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
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8' 
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
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),

                    ), train_eps=False))
                else:
                    self.layers.append(GINConv(
                    nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
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
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index,edge_attr=edge_attr, edge_atten=edge_atten)
            z = self.activation(z)
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
class modeler_clustering(nn.Module):
    def __init__(self, args,nb_classes):
        super(modeler_clustering, self).__init__()
        self.args = args
        self.n_clusters = nb_classes

        self.alpha = 1.0

        self.center = nn.Parameter(torch.FloatTensor(self.n_clusters, self.args.hidden))


        self.init_weight()

    def init_weight(self):

        nn.init.xavier_normal_(self.center)


    def forward(self, H1):

        t = H1.unsqueeze(1)
        y = t.repeat(1, self.n_clusters, 1)
        M1 = self.center.unsqueeze(0)
        M2 = M1.repeat(H1.shape[0], 1, 1)
        dis = ((torch.sum(torch.pow(y - M2, 2), dim=-1) / self.alpha + 1.0) ** (-(self.alpha + 1) / 2))

        Q = (dis.t() / torch.sum(dis, 1)).t()

        return Q,dis
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
            nn.Dropout(p=0.2),
            nn.Linear(2*hidden_dim, hidden_dim),
            InstanceNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class FC_sub(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(FC_sub, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)  


    def forward(self, x):
        a= torch.tanh(self.l1(x))
        b=self.l2(a)
        return b

def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att
class Encoder(torch.nn.Module):
    def __init__(self, gcn1,mlp1,mlp2,mlp3,mlp4,context,objects,bno,bnc):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1

        self.mlp1 = mlp1
        self.mlp_IB = mlp2
        self.mlp3 = mlp3
        self.mlp4 = mlp4
        self.context = context
        self.objects = objects

        self.bno = bno
        self.bnc = bnc
        self.mlp_co = FC(input_dim=64, hidden_dim=64, output_dim=2)
        self.mlp_o = FC(input_dim=64, hidden_dim=64, output_dim=2)
        self.mlp_co_s = FC(input_dim=64, hidden_dim=64, output_dim=2)
        self.optimizer_co = Adam(self.mlp_co.parameters(),lr=0.005)
        self.optimizer_o = Adam(self.mlp_o.parameters(),lr=0.005)
        self.optimizer_co_s = Adam(self.mlp_co_s.parameters(),lr=0.005)
    def forward(self, data,flag=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
         
        z1, g1 = self.gcn1(x, edge_index, batch)
       
        IB_score = self.mlp_IB(z1)
        assignment = F.softmax(IB_score,dim=1)


        edge_weight_c = lift_node_att_to_edge_att(assignment[:,0], data.edge_index).unsqueeze(1)
        edge_weight_o = lift_node_att_to_edge_att(assignment[:,1], data.edge_index).unsqueeze(1)

        z_M,_= self.context(x, edge_index, batch,edge_attr=None,edge_atten=edge_weight_o)
        z_res,_ = self.objects(x,edge_index, batch,edge_attr=None,edge_atten=edge_weight_c)
        
        g_res =global_add_pool(z_res, batch)
        g_M = global_add_pool(z_M, batch)
        
        num = g_res.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        g_co = g_res[random_idx] + g_M 
        g_co_s = g_res+ g_M[random_idx]

        h_co =self.mlp_co(g_co)
        h_M = self.mlp_o(g_M)
        h1 = self.mlp1(g1)
        h_res = self.mlp4(g_res)
        h_shuf =self.mlp_co_s(g_co_s)
        y_shuf= data.y[random_idx]

        
        proj_M = self.mlp3(g_M)
        proj1 = self.mlp3(g1) 
        return h1,g1,h_M, g_M,proj1,proj_M,assignment,h_co,h_res,h_shuf,y_shuf


        
def train_M(encoder_model, data, optimizer_mlp3,cur, device):
    encoder_model.train()
    epoch_loss = 0
    CE = nn.CrossEntropyLoss()
    optimizer_mlp3.zero_grad()
   
    data = data.to(device)

    if data.x.shape[1]==0:
        num_nodes = data.batch.size(0)
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

    h1,g1,h_M,g_M,proj1,proj_M,ass,h_co,h_res,_,_ = encoder_model(data,False)

    #----------------------------------------------------#
    con_loss = -1*calc_loss(proj1,proj_M)


    loss = con_loss 

    loss.backward()
    optimizer_mlp3.step()

    return loss

def train(encoder_model, data, optimizer_mlp1,optimizer_gcn1,optimizer_mlp2, optimizer_context,optimizer_objects,optimizer_mlp4,optimizer_bno,optimizer_bnc,device):
    encoder_model.train()
    epoch_loss = 0
    CE = nn.CrossEntropyLoss()
    
    data = data.to(device)
    optimizer_mlp1.zero_grad()
    optimizer_gcn1.zero_grad()
    optimizer_mlp2.zero_grad()
    optimizer_context.zero_grad()
    optimizer_objects.zero_grad()
  
    optimizer_mlp4.zero_grad()
    optimizer_bno.zero_grad()
    optimizer_bnc.zero_grad()
    encoder_model.optimizer_o.zero_grad()
    encoder_model.optimizer_co.zero_grad()
    encoder_model.optimizer_co_s.zero_grad()

    if data.x.shape[1]==0:
        num_nodes = data.batch.size(0)
        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
    
    h1,g1,h_M,g_M,proj1,proj_M,ass,h_co,h_res,h_shuf,y_shuf = encoder_model(data,True)

    ce = CE(ass,ass)
    con_loss = calc_loss(proj1,proj_M)
    cls_loss0 = CE(h1,data.y)
    cls_loss1 = CE(h_M,data.y)
    cls_loss4= CE(h_co,data.y)
    uniform_target = torch.ones_like(h_res, dtype=torch.float).to(device) / h_res.shape[1]

    cls_loss2 = CE(h_res,data.y)
    cls_loss3 = CE(h_shuf,y_shuf)
    v=torch.ones(data.edge_index.shape[1]).cuda()
    adj = torch.sparse_coo_tensor(data.edge_index, v, (data.batch.size(0), data.batch.size(0))).to_dense().cuda()
    lian = torch.matmul(torch.matmul(ass.T,adj),ass)
    norm_lian = F.normalize(lian, p=1, dim=1)

    coc_loss = torch.norm(norm_lian-torch.eye(2).cuda(),p='fro')/(torch.max(data.batch)+1)

    
    loss = 2*cls_loss1 + 0.1 *con_loss + coc_loss + 1*cls_loss0 + 0.5*cls_loss3 + 1*cls_loss2
    loss.backward()
    optimizer_mlp1.step()
    optimizer_gcn1.step()
    optimizer_mlp2.step()
    optimizer_mlp4.step()
    
    optimizer_context.step()
    optimizer_objects.step()
    optimizer_bno.step()
    optimizer_bnc.step()
    encoder_model.optimizer_o.step()
    encoder_model.optimizer_co.step()
    encoder_model.optimizer_co_s.step()

    return loss

def test(encoder_model, dataloader, device):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x.shape[1]==0:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        h1, g1,h_M,g_M,_,_,ass,_,_,_,_= encoder_model(data)
        x.append(h_M)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    a=torch.count_nonzero(y)
    print(a)
    print(y.shape)
    out = torch.argmax(x, dim=1)

    acc=  accuracy_score(y.detach().cpu().numpy(), out.detach().cpu().numpy())
    F1 = f1_score( y.detach().cpu().numpy(), out.detach().cpu().numpy(), average="micro")
    return acc, F1
def val(encoder_model, dataloader, device):
    encoder_model.eval()
    CE = nn.CrossEntropyLoss()
    x = []
    y = []
    loss=0
    for data in dataloader:
        data = data.to(device)
        if data.x.shape[1]==0:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        h1, g1,h_M,g_M,_,_,ass,_,_,_,_= encoder_model(data)
        cls_loss = CE(h_M,data.y).item()
        loss+=cls_loss
        x.append(h_M)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    out = torch.argmax(x, dim=1)
    acc=  accuracy_score(y.detach().cpu().numpy(), out.detach().cpu().numpy())
    F1 = f1_score( y.detach().cpu().numpy(), out.detach().cpu().numpy(), average="micro")
    return acc,loss/len(dataloader.dataset)
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
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--dataset', type=str, default='PROTEINS') 
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--inner_steps', type=int, default=1)
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

    graphs = load_dataset(dataset_name=args.dataset)
    acc_avg=[]
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(graphs, args.folds))):
        print(f"============================{fold+1}/{args.folds}==================================")
        train_dataset = graphs[train_idx]
        test_dataset = graphs[test_idx]
        val_dataset = graphs[val_idx]
        dataloader = DataLoader(train_dataset, batch_size=128,shuffle=True) 

        val_dataloader = DataLoader(val_dataset, batch_size=128,shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=128,shuffle=False) 
        gcn1 = GConv(gnn='GIN',input_dim=max(graphs.num_features, 1), hidden_dim=64, num_layers=2).to(device)

        mlp1 = FC(input_dim=64, hidden_dim=64, output_dim=graphs.num_classes)
        mlp2 = FC_sub(input_dim=64, hidden_dim=64, output_dim=2)
        mlp3 = FC(input_dim=64, hidden_dim=64, output_dim=64)
        mlp4 = FC(input_dim=64, hidden_dim=64, output_dim=64)
        bnc = BatchNorm1d(64)
        bno= BatchNorm1d(64)

        context = GConv(gnn='GIN',input_dim=max(graphs.num_features, 1), hidden_dim=64, num_layers=2)
        objects = GConv(gnn='GIN',input_dim=max(graphs.num_features, 1), hidden_dim=64, num_layers=2)

        encoder_model = Encoder(gcn1=gcn1,mlp1=mlp1,mlp2=mlp2,mlp3=mlp3,mlp4=mlp4,context= context,objects=objects,bno=bno,bnc=bnc).to(device)
        
        optimizer_gcn1 = Adam(gcn1.parameters(),lr=0.001)
        optimizer_mlp1 = Adam(mlp1.parameters(),lr=0.005)
        optimizer_mlp2 = Adam(mlp2.parameters(),lr=0.001)
        optimizer_mlp3 = Adam(mlp3.parameters(),lr=0.001)
        optimizer_mlp4 = Adam(mlp4.parameters(),lr=0.005)
        optimizer_bnc = Adam(bnc.parameters(), lr=0.01)
        optimizer_bno = Adam(bno.parameters(), lr=0.01)
        
        optimizer_context = Adam(context.parameters(),lr=0.001)
        optimizer_objects = Adam(objects.parameters(),lr=0.001)

        best_acc=0
        best_loss=1e10
        best_val_epoch=0
        for epoch in range(1, args.epoch+1):
          cur = np.log(1 + args.temp_r * epoch)
          cur = min(max(0.05, cur), 0.5)
          for data in dataloader:
            for i in range(int(args.inner_steps)):
                loss = train_M(encoder_model, data, optimizer_mlp3,cur, device)
            for i in range(int(args.outer_steps)):
                loss = train(encoder_model, data, optimizer_mlp1,optimizer_gcn1,optimizer_mlp2,optimizer_context,optimizer_objects,optimizer_mlp4,optimizer_bnc,optimizer_bno,device)
          val_acc,val_loss = val(encoder_model, val_dataloader, device)

          if val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder_model)
        best_model.eval()
        with torch.no_grad():
            acc,F1 = test(best_model, test_dataloader, device)
            acc_avg.append(acc)
            print(f'(E): Test accuracy={acc:.4f}')

    print(np.mean(np.array(acc_avg)))
        
if __name__ == '__main__':
    main()
