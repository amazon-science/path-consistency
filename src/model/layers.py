"""
modified from https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/roi_heads/association_head.py
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
        dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        if self.num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(
                nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            if self.dropout > 0.0:
                self.dropouts = nn.ModuleList(
                    nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1 and self.dropout > 0.0:
                x = self.dropouts[i](x)
        return x

class ATTWeightHead(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers, dropout, share_kq=False):
        super().__init__()
        self.q_proj = MLP(input_dim, feature_dim, feature_dim, 
                    num_layers, dropout)
        if share_kq:
            self.k_proj = self.q_proj
        else:
            self.k_proj = MLP(input_dim, feature_dim, feature_dim, 
                        num_layers, dropout)

    def forward(self, query, key):
        '''
        Inputs:
          query: B x T x N x D
          key: B x T x N x D
        '''
        k = self.k_proj(key) 
        q = self.q_proj(query) 
        attn_weights = torch.einsum('btnd,brmd->btnrm', q, k) # B, T, N, T, N+1

        return attn_weights

    def forward_for_da(self, query, key):
        '''
        Inputs:
          query: B x T x N x D
          key: B x T x N+1 x D
        '''
        k = self.k_proj(key) 
        q = self.q_proj(query) 
        attn_weights = torch.einsum('btnd,btmd->btnm', q, k) # B, T, N, N+1

        return attn_weights

    def forward_for_buffer(self, query, key):
        '''
        Inputs:
          query: T x N x D
          key: T x N x D
        '''
        k = self.k_proj(key) 
        q = self.q_proj(query) 
        attn_weights = torch.einsum('tnd,rmd->tnrm', q, k) # T, N, T, N+1

        return attn_weights
