import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import GlobalAttentionPooling, MaxPooling, AvgPooling
import math
from copy import deepcopy
from random import randint, choice

class TBCNNCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(TBCNNCell, self).__init__()
        self.W_left=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_right=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.W_top=nn.Parameter(torch.rand(x_size, h_size), requires_grad=True)
        self.b_conv = nn.Parameter(torch.zeros(1, h_size), requires_grad=True)

        self.W_left.data.uniform_(-0.1, 0.1)
        self.W_right.data.uniform_(-0.1, 0.1)
        self.W_top.data.uniform_(-0.1, 0.1)
        self.b_conv.data.uniform_(-0.1, 0.1)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        child_nums=nodes.mailbox['h'].size()[1]
        if child_nums==1:
            #W_child=(self.W_left+self.W_right)/2
            #c_s=torch.matmul(nodes.mailbox['h'],W_child)
            c_s=torch.matmul(nodes.mailbox['h'],self.W_left)
            children_state=c_s.squeeze(1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        else:
            left_weight=[(child_nums-1-i)/(child_nums-1) for i in range(child_nums)]
            right_weight=[i/(child_nums-1) for i in range(child_nums)]
            left_weight=torch.tensor(left_weight).to(self.W_left.device)
            right_weight=torch.tensor(right_weight).to(self.W_left.device)

            child_h=nodes.mailbox['h']
            
            child_h_left=torch.matmul(child_h,self.W_left)
            left_weight=left_weight.unsqueeze(1).unsqueeze(0)
            child_h_left=child_h_left*left_weight

            child_h_right=torch.matmul(child_h,self.W_right)
            right_weight=right_weight.unsqueeze(1).unsqueeze(0)
            child_h_right=child_h_right*right_weight
            
            children_state=child_h_left+child_h_right
            children_state=children_state.sum(dim=1)
        '''child_nums=nodes.mailbox['h'].size()[1]
        if child_nums==1:
            W_child=(self.W_left+self.W_right)/2
            c_s=torch.matmul(nodes.mailbox['h'],W_child)
            #c_s=torch.matmul(nodes.mailbox['h'],self.W_left)
            children_state=c_s.squeeze(1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)
        else:
            W_children=[]
            for i in range(child_nums):
                W_children.append((i/(child_nums-1))*self.W_right+((child_nums-1-i)/(child_nums-1))*self.W_left)
            W_s=torch.stack(W_children,dim=0)
            child_h=nodes.mailbox['h'].unsqueeze(-2) #size(batch, child_nums, 1, x_size)
            children_state=torch.matmul(child_h,W_s)
            children_state=children_state.squeeze(-2)
            children_state=children_state.sum(dim=1)
            h=torch.relu(children_state+torch.matmul(nodes.data['h'],self.W_top)+self.b_conv)'''
        return {'h': h}

    def apply_node_func(self, nodes):
        return {'h': nodes.data['h']}

class TBCNNClassifier(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 n_classes,
                 vocab_size,
                 num_layers=1):
        super(TBCNNClassifier, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        self.layers=nn.ModuleList(TBCNNCell(x_size, h_size) for _ in range(num_layers))
        self.embeddings=nn.Embedding(vocab_size,x_size)
        self.classifier=nn.Linear(h_size,n_classes)
        #self.pooling=GlobalAttentionPooling(nn.Linear(h_size,1))
        self.pooling=MaxPooling()
        #self.pooling=AvgPooling()

    def forward(self, batch,root_ids=None):
        batch.ndata['h']=self.embeddings(batch.ndata['type'])
        for i in range(self.num_layers):
            batch.update_all(message_func=self.layers[i].message_func,
                            reduce_func=self.layers[i].reduce_func,
                            apply_node_func=self.layers[i].apply_node_func)
        batch_pred=self.pooling(batch,batch.ndata['h'])
        batch_pred=self.classifier(batch_pred)
        return batch_pred
