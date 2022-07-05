import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GraphConv, EdgeConv, GATConv

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x):
        # x = self.layer1(x)
        # if self.use_bn:
        #     x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        # self.act = nn.PReLU() if act is not None else None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=1.414)
            # if m.bias is not None:
            #     nn.init.xavier_uniform_(m.bias.data, gain=1.414)

 #   Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj = None, batch=False):
        seq_fts = self.fc(seq)
        if adj is not None:
            if batch:
                out = torch.bmm(adj, seq_fts) + seq_fts
            else:
                out = torch.mm(adj, seq_fts) + seq_fts
        else:
            out = seq_fts
        if self.bias is not None:
            out += self.bias
        # if self.act:
        #     out = self.act(out)
        return out #, out /out.norm(dim=2)[:, :,None]

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout = 0.1, alpha = 0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, do_att=True, do_batch = False):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if do_batch:
            h_prime = torch.bmm(adj, Wh) + Wh
            if self.concat:
                return F.elu(h_prime)
            else:
                return h_prime

        else:
                if do_att:
                    a_input = self._prepare_attentional_mechanism_input(Wh)
                    e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

                    zero_vec = -9e15 * torch.ones_like(e)
                    attention = torch.where(adj > 0, e, zero_vec)
                    attention = F.softmax(attention, dim=1)
                    attention = F.dropout(attention, self.dropout, training=self.training)
                else:
                    attention = adj
                h_prime = torch.matmul(attention, Wh) + Wh

                if self.concat:
                    return F.elu(h_prime), attention#.detach()
                else:
                    return h_prime, attention#.detach()

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout = 0):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.out_att = GraphAttentionLayer(nfeat , nhid, dropout=0.1, alpha=0.1, concat=True)

    def forward(self, x, adj, do_att, do_batch):
        X, A = self.out_att(x, adj, do_att, do_batch)
        return X, A


class MyGAT(nn.Module):
    def __init__(self, nin, cfg):
        super(MyGAT, self).__init__()
        self.layer1 = nn.Linear(nin, cfg[0])
        self.MyGAT_layer_1 = GAT(cfg[0], cfg[1])
        self.MyGAT_layer_2 = GAT(cfg[1], cfg[-1])
        self.activation = F.elu

    def forward(self, x, A):
        x = F.dropout(x, 0.2, training=self.training)
        x = self.layer1(x)
        # x = self.bn_0(x)
        x, _ = self.MyGAT_layer_1(self.activation(x), A, True, False)
        # x = self.bn_1(x)
        x, S = self.MyGAT_layer_2(self.activation(x), A, True, False)
        # x = self.bn_2(x)
        feature = x
        return feature, S

class MyGAT_view(nn.Module):
    def __init__(self, nin, cfg, cls_number = 2, view = 5):
        super(MyGAT_view, self).__init__()
        self.MyGCN_1 = MyGAT(nin, cfg)
        self.view = view
        self.cls = nn.Linear(cfg[-1], cls_number)
    def forward(self, x, A):
        feature_list = []
        S_list = []
        for x_i, A_i in zip(x, A):
            feature_0, S_0 = self.MyGCN_1(x_i, A_i)
            feature_list.append(feature_0)
            S_list.append(S_0)
        feature =  torch.stack(feature_list).mean(dim=0)
        S = torch.stack(S_list).mean(dim=0)
        # out = self.cls(feature)
        return feature, S

class MyGCN(nn.Module):
    def __init__(self, nin, cfg, cls_number = 2):
        super(MyGCN, self).__init__()
        self.layer1 = nn.Linear(nin, cfg[0])
        self.MyGCN_1 = GCN(cfg[0], cfg[1])
        self.MyGCN_2 = GCN(cfg[1], cfg[-1])
        self.MyGCN_3 = GCN(cfg[1], cfg[-1])
        self.cls = nn.Linear(cfg[-1], cls_number)
        self.activation = F.elu

    def forward(self, x, A, S):

        bnsize = x.size(0)
        brain_size =  x.size(1)
        x = F.dropout(x, 0.2, training=self.training)
        x = self.layer1(x)
        x = self.activation(x)

        x_1 = self.MyGCN_1(x, A, batch = True)
        x_1 = self.activation(x_1)
        x_2 = self.MyGCN_2(x_1, A, batch = True)
        x_2 = self.activation(x_2)
        feature = x_2.mean(dim=1)  #x_2.mean(dim=1) torch.max(x,dim=1).values
        feature = feature.reshape(bnsize, feature.size(-1))
        feature = self.MyGCN_3(feature, S)
        return feature


class MyGCN_view(nn.Module):
    def __init__(self, nin, cfg, cls_number = 2, view = 4):
        super(MyGCN_view, self).__init__()
        self.MyGCN_1 = MyGCN(nin, cfg, cls_number = 2)
        self.view = view
        self.cls = nn.Linear(cfg[-1], cls_number)
    def forward(self, x, A, A_181):
        feature_list = []
        for x_i, A_i in zip(x, A ):
            feature_0, out_0 = self.MyGCN_1(x_i, A_i, A_181)
            feature_list.append(feature_0)
        feature =  torch.stack(feature_list).mean(dim=0)
        out = self.cls(feature)
        return feature, out


class MyGHU(nn.Module):
    def __init__(self, nb_dim_up, nb_dim_down, cfg_up, cfg_down, view,cls_number = 2):
        super(MyGHU, self).__init__()
        self.Uper_model = MyGAT_view(nb_dim_up, cfg_up, cls_number)
        self.Down_model = MyGCN_view(nb_dim_down, cfg_down, cls_number, view)
        self.cls = nn.Linear(cfg_up[-1]+cfg_down[-1], cls_number)
        self.a = torch.Parameter(torch.Tensor(1))
        self.b = torch.Parameter(torch.Tensor(1))
        nn.init.ones_(self.a)
        nn.init.ones_(self.b)
    def forward(self, data_feature_up, data_graph_up, data_feature, data_graph):
        Z_mlp, S = self.Uper_model(data_feature_up, data_graph_up)
        Z_g, P_g = self.Down_model(data_feature, data_graph, S)
        out_class = self.cls(torch.cat((self.a * Z_mlp, self.b * Z_g), dim=1))
        return out_class
