import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        # weight
        self.W_layer = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W_layer.weight, gain=1.414)

        dim = 1
        self.a_layer1 = nn.Linear(out_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.a_layer1.weight, gain=1.414)

        self.a_layer2 = nn.Linear(out_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.a_layer2.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W_layer(h)

        Wh1 = self.a_layer1(Wh)
        Wh2 = self.a_layer2(Wh)

        att = self.leakyrelu(torch.mm(Wh1, Wh2.T))

        att_exp = torch.exp(att)

        att_adj = torch.where(adj > 0, att_exp, torch.zeros_like(att_exp))
        att_adj = att_adj / att_adj.sum(dim=1, keepdim=True)
        att_dropout = F.dropout(att_adj, self.dropout, training=self.training)
        out = torch.matmul(att_dropout, Wh)

        if self.concat:
            return F.elu(out)
        else:
            return out

    def __repr__(self):
        return  self.__class__.__name__ + ' (' + '\n' + \
        f'in_dim: {self.in_dim},' + '\n' + f'out_dim: {self.out_dim},' + '\n' + f'dropout: {self.dropout},' \
        + '\n' + f'alpha: {self.alpha},' + '\n' + ')'


class GraphAttnNetwork(nn.Module):
    def __init__(self, in_dim, c_dim, nclass, dropout, alpha, num_gals_in, num_gals_predict):
        super().__init__()
        self.dropout = dropout

        self.parallel_attentions = nn.ModuleList([GraphAttentionLayer(in_dim, c_dim, dropout=dropout, alpha=alpha, concat=True)
                                         for _ in range(num_gals_in)])

        self.predicts_attentions = nn.ModuleList([GraphAttentionLayer(c_dim * num_gals_in, nclass, dropout=dropout, alpha=alpha, concat=False)
                                       for _ in range(num_gals_predict)])

    def forward(self, x, adj):
        # dropout
        x_dropout = F.dropout(x, self.dropout, training=self.training)

        # parallel attentions
        x_parallel = [att(x_dropout, adj) for att in self.parallel_attentions]
        x_parallel = torch.cat(x_parallel, dim=1)

        # dropout
        x_parallel = F.dropout(x_parallel, self.dropout, training=self.training)

        # predict attentions
        x_predicts = [att(x_parallel, adj) for att in self.predicts_attentions]
        x_predicts_act = [F.elu(predict) for predict in x_predicts]
        x_predicts_softmax = [F.log_softmax(predict_act, dim=1) for predict_act in x_predicts_act]
        return x_predicts_softmax

