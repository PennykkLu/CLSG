# /usr/bin/env python36
# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_lp_pool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_lp_pool = use_lp_pool
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):

        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads) # WQ
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.heads)  # WK batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads) # WV
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_lp_pool == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.last_layernorm(a)
        return a, alpha


class AttMix(Module):
    def __init__(self, default_opt,model_opt, n_node):
        super(AttMix, self).__init__()
        self.hidden_size = model_opt.hidden_size
        self.n_node = n_node
        self.norm = model_opt.norm
        # self.scale = model_opt.scale
        self.batch_size = default_opt.batch_size
        self.heads = model_opt.heads
        self.use_lp_pool = model_opt.use_lp_pool
        self.softmax = model_opt.softmax
        self.dropout = model_opt.dropout
        self.last_k = model_opt.last_k
        self.dot = model_opt.dot
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.l_p = model_opt.l_p
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_lp_pool=self.use_lp_pool) # EQ.(7)
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k): # W of EQ.(6)
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def compute_scores(self, hidden, mask):

        hts = []

        lengths = torch.sum(mask, dim=1)

        for i in range(self.last_k): # EQ.(6)
            hts.append(self.linear_q[i](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))

        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]

        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)

        hidden1 = hidden
        hidden = hidden1[:, :mask.size(1)]

        ais, weights = self.mattn(hts, hidden, mask) # EQ.(7)
        a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1)) # sessions

        b = self.embedding.weight[1:] # items

        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        # scores = torch.matmul(a, b.transpose(1, 0))
        # if self.scale:
        #     scores = 16 * scores
        # return scores
        return a,b.transpose(1, 0)

    def forward(self, seqs,length):
        alias_inputs, A, items, mask, mask1, n_node = self.process_data(seqs)
        alias_inputs = alias_inputs.to(seqs.device)
        A = A.to(seqs.device)
        items = items.to(seqs.device)
        mask = mask.to(seqs.device)
        mask1 = mask1.to(seqs.device)
        n_node = n_node.to(seqs.device)
        hidden = self.embedding(items)
        # assert not torch.isnan(hidden).any()
        hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)
        hidden = F.dropout(hidden, 0.1, training=self.training)
        seq_hidden = torch.stack([self.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        seq_shape = list(seq_hidden.size())
        seq_hidden = seq_hidden.view(-1, self.hidden_size)
        norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
        seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
        seq_hidden = seq_hidden.view(seq_shape)
        s_h, all_item_embedding = self.compute_scores(seq_hidden,mask)
        return s_h, all_item_embedding

    def process_data(self,seqs,order=2):
        inputs_raw,mask_raw = [],[]
        items, n_node, A, masks, alias_inputs = [], [], [], [], []
        len_max = seqs.shape[1]
        for u_input in seqs:
            n = int(torch.count_nonzero(u_input))
            mask_raw.append([1] * n + [0] * (len_max - n))
            u_input = u_input.tolist() + [0] * (len_max - len(u_input))
            inputs_raw.append(np.array(u_input))
            n_node.append(n+1)
        mask_raw = np.array(mask_raw)
        max_n_node = np.max(n_node)
        l_seq = np.sum(mask_raw, axis=1)
        max_l_seq = mask_raw.shape[1]
        max_n_node_aug = max_n_node
        for k in range(order - 1):
            max_n_node_aug += max_l_seq - 1 - k
        for idx, u_input in enumerate(inputs_raw):
            node = np.array(np.unique(u_input).tolist() + [0])
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node_aug, max_n_node_aug))
            mask1 = np.zeros(max_n_node_aug)
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    if i == 0:
                        mask1[0] = 1
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                mask1[u] = 1
                mask1[v] = 1
                u_A[u][v] += 1

                for t in range(order - 1):
                    if i == 0:
                        k = max_n_node + t * max_l_seq - sum(list(range(t + 1))) + i
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        k = max_n_node + t * max_l_seq - sum(list(range(t + 1))) + i + 1
                        u_A[u][k] += 1
                        u_A[k - 1][k] += 1
                        mask1[k] = 1
                    if i < l_seq[idx] - t - 2:
                        # l = np.where(node == u_input[i + t + 2])[0][0]
                        l = np.where(node == u_input[i + t + 2])[0]
                        if len(l):
                        # if l is not None and l > 0:
                            u_A[k - 1][l[0]] += 1
                            mask1[l[0]] = 1
                            # u_A[k - 1][l] += 1
                            # mask1[l] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            masks.append(mask1)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.tensor(alias_inputs).long()
        A = torch.tensor(A).float()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask_raw).long()
        mask1 = torch.tensor(masks).long()
        n_node = torch.tensor(n_node).long()
        return alias_inputs, A, items, mask, mask1, n_node