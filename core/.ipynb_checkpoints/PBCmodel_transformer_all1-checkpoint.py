import torch
import torch.nn as nn
import torch.nn.functional as F
from .plastic_cls import PlasticModel
from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine
import math

#DeepEC(CNN) + Transformer 
class PlasticTransformer(nn.Module):
    def __init__(self, kernel_size, hidden_dim=384,  n_class = 14):
        super(PlasticTransformer, self).__init__()
        
        self.backbone = PlasticModel(kernel_size = kernel_size)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats = hidden_dim // 2, normalize = True, maxH = 1, maxW = 1)
        self.transformer = Transformer(d_model = hidden_dim, dim_feedforward = int(hidden_dim * 4))
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(n_class, hidden_dim)
        self.fc = GroupWiseLinear(n_class, hidden_dim)
        
    def forward(self, data):
        feat1, feat2, feat3 = self.backbone(data)
        feat = torch.cat([feat1, feat2, feat3], dim = 1)
        pos = self.position_encoding(feat)
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(feat), query_input, pos)[0] # B,K,d
        out = self.fc(hs[-1])
        return out
        
        
class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
