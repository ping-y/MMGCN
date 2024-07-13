import torch
import torch.nn.functional as F
import snf
import model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from _0_get_impr_type import get_impr_type
import math
from typing import Tuple
import logging
from torch import Tensor, nn
from typing import List, Optional
import os
import pandas as pd
import numpy as np
import pickle
os.environ["DGLBACKEND"] = "pytorch"
cuda = True if torch.cuda.is_available() else False
logger = logging.getLogger(__name__)


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = np.sum(count) / (count[i] + 1)
    return sample_weight


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def stats_type_flag_adj(type_flag_adj):
    dic=dict()
    for i in range(1,8):
        is_i=np.where(type_flag_adj==i, 1, 0)
        count_i=is_i.sum().sum()
        dic[i]=count_i
    print('dic')
    print(dic)



def maxmin_scale(df):
    for col in list(df.columns):
        if df[col].sum() == df.shape[0] or df[col].sum() == 0:
            del df[col]
        else:
            df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    return df


def get_W(x, thres_type, threshold=0.00):  
    affinity_networks = snf.make_affinity(x, metric='euclidean', K=20, mu=0.5)
    adj_hat = torch.FloatTensor(snf.snf(affinity_networks, K=20))
    adj_hat[adj_hat < threshold] = 0

    type_flag_adj=get_impr_type(affinity_networks,thres_type)
    type_flag_adj=np.where(adj_hat>0, type_flag_adj, 0)
    stats_type_flag_adj(type_flag_adj)

    adj_hat=adj_hat.numpy()
    return adj_hat, type_flag_adj


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_static=None

    def step(self, acc, model, max_is_better=True):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            if max_is_better==True:
                if score < self.best_score:
                    self.counter += 1
                    if self.counter%5==0:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(model)
                    self.counter = 0
            else:
                if score > self.best_score:
                    self.counter += 1
                    if self.counter % 5 == 0:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(model)
                    self.counter = 0
        return self.early_stop,self.best_score, self.model_static

    def save_checkpoint(self, model):
        self.model_static=model


class GCGCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCGCN, self).__init__()
        self.gc1 = model.GraphConvolution(n_in, n_hid)
        self.gc2 = model.GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(0.5)

    def forward(self, x, adj, attn_weights):
        adj=torch.mul(attn_weights, adj)
        x = self.gc1(x, adj)
        x = F.elu(x)
        x = self.dp1(x)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta0 = torch.softmax(w, dim=0)
        beta = beta0.expand((z.shape[0],) + beta0.shape)
        return (beta * z).sum(1),beta0


class MMGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, ( "Self-attention requires query, key and " "value to be of the same size" )

        self.input_proj = nn.Linear(input_dim, embed_dim, bias=bias)

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layer_norm=nn.LayerNorm(embed_dim)
        self.bef_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, 2, bias=bias)
        self.gcgcn1 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_1 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_2 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_3 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_4 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_5 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_6 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_7 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.gcgcn_8 = GCGCN(input_dim, embed_dim, 2, dropout)
        self.semanticAttention=SemanticAttention(embed_dim)
        self.reset_parameters()
        self.onnx_trace = False


    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0.0)

    def forward(
        self,
        query0,
        type_flag_adj,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:

        query = self.input_proj(query0)

        tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, embed_dim]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1,2))
        assert list(attn_weights.size()) == [self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if before_softmax:
            return attn_weights, v

        attn_weights = attn_weights.mean(dim=0).squeeze(0)

        query_gnn1=self.gcgcn_1(query0, type_flag_adj[1][0], attn_weights)
        query_gnn2=self.gcgcn_2(query0, type_flag_adj[1][1], attn_weights)
        query_gnn3=self.gcgcn_3(query0, type_flag_adj[1][2], attn_weights)
        query_gnn4=self.gcgcn_4(query0, type_flag_adj[1][3], attn_weights)
        query_gnn5=self.gcgcn_5(query0, type_flag_adj[1][4], attn_weights)
        query_gnn6=self.gcgcn_6(query0, type_flag_adj[1][5], attn_weights)
        query_gnn7=self.gcgcn_7(query0, type_flag_adj[1][6], attn_weights)

        stack_att = torch.stack([query_gnn1,query_gnn2,query_gnn3,query_gnn4,query_gnn5,query_gnn6,query_gnn7], dim=1)
        attn,beta0=self.semanticAttention(stack_att)

        attn = self.out_proj(attn)
        return attn, attn_weights, beta0


def prepare(path, view_list, num_class,  set_index, cv_i, dataset,thres_type):
    train_sample, val_sample, test_sample = [
        pd.read_csv(path + 'set_{}/{}_{}.csv'.format(set_index, cv_i, name), index_col=0)['0'].values for name in
        ['tr', 'val', 'te']]

    x = [pd.read_csv(path +'{}.csv'.format(name), index_col=0) for name in view_list]
    y = pd.read_csv(path +'y.csv', index_col=0)

    tr = [i.loc[train_sample].values for i in x]
    trval = [i.loc[np.concatenate((train_sample, val_sample))].values for i in x]
    trvalte = [i.loc[np.concatenate((train_sample, val_sample, test_sample))].values for i in x]

    trvalte_adj, trvalte_type_flag_adj = get_W(trvalte, thres_type=thres_type)
    tr_adj, tr_type_flag_adj = trvalte_adj, trvalte_type_flag_adj
    trval_adj, trval_type_flag_adj = trvalte_adj, trvalte_type_flag_adj

    tr_x = tr[0]
    for i in tr:
        tr_x = np.concatenate((tr_x, i), axis=1)
    trval_x = trval[0]
    for i in trval:
        trval_x = np.concatenate((trval_x, i), axis=1)
    trvalte_x = trvalte[0]
    for i in trvalte:
        trvalte_x = np.concatenate((trvalte_x, i), axis=1)

    tr_x = torch.FloatTensor(tr_x)
    trval_x = torch.FloatTensor(trval_x)
    trvalte_x = torch.FloatTensor(trvalte_x)
    tr_labels = y.loc[train_sample]['0'].values
    val_labels = y.loc[val_sample]['0'].values
    te_labels = y.loc[test_sample]['0'].values

    sample_weight_tr = cal_sample_weight(tr_labels, num_class)
    sample_weight_val = cal_sample_weight(val_labels, num_class)

    tr_labels = torch.LongTensor(tr_labels)
    val_labels = torch.LongTensor(val_labels)
    te_labels = torch.LongTensor(te_labels)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    sample_weight_val = torch.FloatTensor(sample_weight_val)
    if cuda:
        tr_x = tr_x.cuda()
        trval_x = trval_x.cuda()
        trvalte_x = trvalte_x.cuda()
        tr_labels = tr_labels.cuda()
        val_labels = val_labels.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
        sample_weight_val = sample_weight_val.cuda()

    tr_adj = torch.FloatTensor(tr_adj)
    trval_adj = torch.FloatTensor(trval_adj)
    trvalte_adj = torch.FloatTensor(trvalte_adj)
    tr_type_flag_adj = torch.Tensor(tr_type_flag_adj)
    trval_type_flag_adj = torch.Tensor(trval_type_flag_adj)
    trvalte_type_flag_adj = torch.Tensor(trvalte_type_flag_adj)

    dic_x= {'tr_x': tr_x, 'trval_x': trval_x, 'trvalte_x': trvalte_x,
            'tr_adj': tr_adj, 'trval_adj': trval_adj, 'trvalte_adj': trvalte_adj,
            'tr_labels': tr_labels, 'val_labels': val_labels, 'te_labels': te_labels,
            'sample_weight_tr': sample_weight_tr, 'sample_weight_val': sample_weight_val,
             'tr_type_flag_adj': tr_type_flag_adj, 'trval_type_flag_adj': trval_type_flag_adj, 'trvalte_type_flag_adj': trvalte_type_flag_adj,
              }
    return dic_x


def handle_type_to_multi_graph(type_flag_adj, adj):
    type_flag_adj=np.array(type_flag_adj)
    adj=np.array(adj)

    ls_adj_weight=[]
    ls_adj_01=[]
    for i in range(1,8):
        adj_weight=torch.FloatTensor(np.where(type_flag_adj ==i, adj, 0))
        adj_weight=adj_weight-torch.diag(adj_weight.diag())+torch.diag(torch.FloatTensor(adj).diag())
        ls_adj_weight.append(adj_weight)

        adj_01=np.where(type_flag_adj == i, 1, 0)
        ls_adj_01.append(torch.LongTensor(adj_01))

    return ls_adj_01, ls_adj_weight


def filter_edges_usingThred(ls_adj_weight, thred):
    ls_adj_weight = [np.array(i) for i in ls_adj_weight]

    ls_thred=[]
    for i in ls_adj_weight:
        tmp=[]
        for j in i:
            for k in j:
                if k>0:
                    tmp.append(k)
        ls_thred.append(np.quantile(np.array(tmp), thred, interpolation='nearest'))

    ls_train_adj_weight_tmp = []
    for i in range(len(ls_adj_weight)):
        adj = ls_adj_weight[i].copy()
        dia = np.diag(np.diag(adj).copy())

        thred_value = ls_thred[i]
        adj = np.where(adj >= thred_value, adj, 0)
        adj = adj - np.diag(np.diag(adj).copy()) + dia
        ls_train_adj_weight_tmp.append(adj)
    ls_adj_weight = ls_train_adj_weight_tmp
    ls_adj_weight=[torch.FloatTensor(i) for i in ls_adj_weight]
    return ls_adj_weight


def get_edge_pred_loss(attention_train, is_homorphy, tr_labels):
    is_homorphy_train=is_homorphy[:tr_labels.shape[0],:][:, :tr_labels.shape[0]]
    edge_pred = attention_train[:tr_labels.shape[0], :][:, :tr_labels.shape[0]]
    loss_edge_pred = F.binary_cross_entropy_with_logits(edge_pred.reshape(-1), is_homorphy_train.reshape(-1))
    return loss_edge_pred


def get_edge_pred_label(gammar, trvalte_adj, tr_labels, val_labels, te_labels):
    y=np.concatenate([tr_labels, val_labels, te_labels])
    flag_1 = np.matmul(y.reshape(-1, 1), y.reshape(1, -1))
    flag_0 = np.matmul((1 - y).reshape(-1, 1), (1 - y).reshape(1, -1))

    trvalte_adj=np.array(trvalte_adj)

    k=12
    tr_adj = trvalte_adj[:tr_labels.shape[0], :][:, :tr_labels.shape[0]]
    diag_original = np.diag(tr_adj).copy()
    tmp_i = tr_adj - np.diag(diag_original)
    ls_thred_value = []
    for j in tmp_i:
        k_value = sorted(list(j))[-k]
        ls_thred_value.append([k_value] * tmp_i.shape[1])
    ls_thred_value = np.array(ls_thred_value)
    tr_adj = np.where(tmp_i >= ls_thred_value, tmp_i, 0)

    tr_flag_1=flag_1[:tr_labels.shape[0], :][:, :tr_labels.shape[0]]
    tr_flag_0 = flag_0[:tr_labels.shape[0], :][:, :tr_labels.shape[0]]
    label_1_weight=(tr_flag_1*tr_adj).sum()/(tr_flag_1*tr_adj>0).sum()
    label_0_weight = (tr_flag_0 * tr_adj).sum()/ (tr_flag_0 * tr_adj>0).sum()

    is_homorphy = np.where(flag_1 > 0, gammar*1, 0)
    is_homorphy = np.where(flag_0 > 0, gammar*label_0_weight/label_1_weight, is_homorphy)
    is_homorphy = torch.FloatTensor(is_homorphy)
    return is_homorphy


def main():
    ls_k_mean = []
    ls_k_special = []

    thres_type=0.6
    theta=0.55
    lr = 0.001
    epoch = 500
    gammar = 0.1
    weight_decay = 0.035
    num_class = 2
    early_stop=True
    dataset='MetaBric'
    path = 'Data/%s/'%(dataset)
    view_list = ['clinical', 'exp', 'cnv']

    ls_specific = []
    ls_mean = []
    for set_index in range(1,6):
        ls_cv_rst = []
        for cv_i in range(1, 6):
            print('=====%d, %d=====' % (set_index, cv_i))
            dic = prepare(path, view_list, num_class, set_index, cv_i, dataset,thres_type)
            is_homorphy=get_edge_pred_label(gammar, dic['trvalte_adj'], np.array(dic['tr_labels']), np.array(dic['val_labels']), np.array(dic['te_labels']))
            ls_adj_01, ls_adj_weight=handle_type_to_multi_graph(dic['trvalte_type_flag_adj'], dic['trvalte_adj'])
            dic['trvalte_type_flag_adj']=[ls_adj_01, ls_adj_weight, dic['trvalte_type_flag_adj']]

            model_dic = {}
            mmGCN = MMGCN(
                    input_dim= dic['trvalte_x'].shape[1],
                    embed_dim=64,
                    num_heads=4,
                    dropout=0.5,
                    bias=True,
                    self_attention=True
            )
            optim = torch.optim.Adam(mmGCN.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            stopper = EarlyStopping(patience=15)
            for i in range(epoch):
                mmGCN.train()
                all_pre, attention_train,beta_train = mmGCN(dic['trvalte_x'], dic['trvalte_type_flag_adj'])
                train_pre=all_pre[:dic['tr_x'].shape[0]]

                ci_train_loss0 = torch.mean(torch.mul(criterion(train_pre, dic['tr_labels']), dic['sample_weight_tr']))
                loss_edge_pred=get_edge_pred_loss(attention_train, is_homorphy, np.array(dic['tr_labels']))
                ci_train_loss=(1-theta)*ci_train_loss0+theta*loss_edge_pred

                trainval_pre = all_pre[:dic['trval_x'].shape[0]]

                ci_val_loss = torch.mean(torch.mul(criterion(trainval_pre[len(dic['tr_labels']):], dic['val_labels']), dic['sample_weight_val']))

                if early_stop and i>50:
                    is_stop, best_score,model_dic = stopper.step(ci_val_loss.item(), mmGCN.state_dict(), max_is_better=False)
                    if is_stop == True:
                        break

                if i%1==0:
                    print(i,"  train_loss", round(ci_train_loss.item(),5), "  val_loss", round(ci_val_loss.item(),5),
                          [_[0] for _ in beta_train.detach().numpy()],
                          'ci_train_loss', round(ci_train_loss0.item(),5),'loss_edge_pred', round(loss_edge_pred.item(),5))
                ci_train_loss.backward()
                optim.step()
                optim.zero_grad()

            pickle.dump(model_dic, open('models/%s/mdl_set%d_cv%d.pkl' % (dataset, set_index, cv_i), 'wb'))

            mmGCN.load_state_dict(model_dic)
            mmGCN.eval()
            _3 , attention_test,beta_test = mmGCN(dic['trvalte_x'], dic['trvalte_type_flag_adj'])

            prediction_3 = _3.argmax(1)[-len(dic['te_labels']):]
            f1=f1_score(dic['te_labels'], prediction_3)
            rec=recall_score(dic['te_labels'], prediction_3)
            prec=precision_score(dic['te_labels'], prediction_3)
            acc=accuracy_score(dic['te_labels'], prediction_3)
            auc= roc_auc_score(dic['te_labels'], torch.softmax(_3, dim=1)[-len(dic['te_labels']):, -1].detach().numpy())

            ls_cv_rst.append(['cv_' + str(cv_i)] + [f1, rec, prec, acc, auc])
            print('ls_cv_rst', ls_cv_rst)

        cv_rsts = pd.DataFrame(ls_cv_rst, columns=['cv', 'test_f1', 'test_recall', 'test_prec', 'test_acc', 'test_auc'])
        cv_rsts['set'] = set_index
        ls_specific.append(cv_rsts)

        ls_mean.append([set_index] + list(cv_rsts[['test_f1', 'test_recall', 'test_prec', 'test_acc', 'test_auc']].mean(axis=0)))
        print('ls_mean', ls_mean)

    ls_specific = pd.concat(ls_specific, axis=0)

    ls_mean = pd.DataFrame(ls_mean, columns=['set', 'test_f1', 'test_recall', 'test_prec', 'test_acc', 'test_auc'])
    ls_mean = ls_mean.append({'test_f1': ls_mean['test_f1'].mean(), 'test_recall': ls_mean['test_recall'].mean(),
                              'test_prec': ls_mean['test_prec'].mean(),
                              'test_acc': ls_mean['test_acc'].mean(), 'test_auc': ls_mean['test_auc'].mean(),
                              'set': 'mean'}, ignore_index=True)
    ls_mean = ls_mean.append({'test_f1': ls_mean['test_f1'].std(), 'test_recall': ls_mean['test_recall'].std(),
                              'test_prec': ls_mean['test_prec'].std(),
                              'test_acc': ls_mean['test_acc'].std(), 'test_auc': ls_mean['test_auc'].std(),
                              'set': 'std'}, ignore_index=True)

    ls_k_mean.append(ls_mean)
    ls_k_special.append(ls_specific)
    pd.concat(ls_k_mean, axis=0).to_excel('results/MMGCN_rst_mean_std.xlsx')


if __name__ == "__main__":
    main()
