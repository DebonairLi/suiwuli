#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable

global count
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        # hy = newgate + inputgate * (hidden - newgate)
        hy = hidden * (1-inputgate) + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.b2 = nn.Embedding(self.n_node, 1)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.n_v = opt.n_v
        self.n_h = opt.n_h
        self.dataset = opt.dataset

        #drop
        self.drop_ratio = opt.drop
        self.dropout = nn.Dropout(self.drop_ratio)

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_ones = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

        self.feature_fate=nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
          
#         s = torch.sum(mask, 1).float().reshape(mask.shape[0],-1)
#         avg_hidden = 1/s * torch.sum(hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
#         avg_hidden = avg_hidden.view(hidden.shape[0],-1,hidden.shape[2])
#         print(avg_hidden.shape)

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        
#         s = torch.sum(mask, 1).float().reshape(mask.shape[0],-1)
#         q3 = 1/s * torch.sum(hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
#         q3 = q3.view(hidden.shape[0],-1,hidden.shape[2])
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        #feature gating
        avg_hidden = a.view(ht.shape[0],1,ht.shape[1])
        feature_gate = hidden
        gate=torch.sigmoid(self.feature_fate(feature_gate + avg_hidden))
        gated = feature_gate*gate#[100,69,100]
#         print(hidden.shape)
        
#         print(a.shape)
        
        out = torch.cat([a,ht],1)
 
        out = self.dropout(out)

        z = self.fc(out) #[batch_size, hidden_size]


        b = self.embedding.weight[1:]  # n_nodes x latent_size
        
        
#         gated = gated.transpose(1,2)
#         feature_gate = F.max_pool1d(gated,gated.size(2)).squeeze(2)
        gated = torch.matmul(gated, b.transpose(1,0)).transpose(1,2)
        gated = F.max_pool1d(gated, gated.size(2)).squeeze(2)

        scores = torch.matmul(z, b.transpose(1, 0)) + gated
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs) # batch_size x seq_length x latent_size
        hidden = self.gnn(A, hidden) # batch_size x seq_length x latent_size
        return hidden

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long()) #100*69 69应为序列最大长度
    items = trans_to_cuda(torch.Tensor(items).long()) #batchsize * seq_length
    A = trans_to_cuda(torch.Tensor(A).float()) #batchsize*seq_length*2(seq_length)
    mask = trans_to_cuda(torch.Tensor(mask).long()) #100*69
    hidden = model(items, A) # batch_size x seq_length x latent_size 调用128行代码
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # print("***********",seq_hidden.shape) #100*69*100
    return targets, model.compute_scores(seq_hidden, mask) #targets.shape (100,)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    #each slices correspond to each batch
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()#清空上一步的残余更新值
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        # print(targets.shape, scores.shape)
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.data.cpu()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    Count = []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        # print("111111111111111111")

        targets, scores = forward(model, i, test_data)

        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                Count.append(np.where(score == target - 1)[0][0] + 1)
    hits = np.mean(hit) * 100
    mrrs = np.mean(mrr) * 100
    print("len_test_seqs", len(mrr))
    s = 0
    zhong = 0
    for i in range(20):
        if i+1 in Count:
            if s < Count.count(i+1):
                s = Count.count(i+1)
                zhong = i+1

    half = 0
    for i in range(9,20):
        if i+1 in Count:
            half += Count.count(i+1)

    print("zhong shu",zhong, s, s * 1.0 / len(mrr))
    print("1_count", Count.count(1), Count.count(1) * 1.0 / len(mrr))
    print("20_count",Count.count(20))
    print("median", np.median(Count), Count.count(np.median(Count)), Count.count(np.median(Count)) * 1.0 / len(mrr) )
    print("mean", np.mean(Count))
    print("half", half)

    return hits, mrrs

