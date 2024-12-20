import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.parametrize as parametrize


class EquivariantP(nn.Module):
    def __init__(self, in_group, out_group, dtype=torch.float):
        super().__init__()
        self.in_group = in_group
        self.out_group = out_group
        self.dtype = dtype

    def forward(self, X):
        weights = torch.zeros_like(X).to(self.dtype) 

        for g,h in zip(self.in_group.transformations, self.out_group.transformations):
            h_1 = torch.inverse(h).to(self.dtype) #TODO: if this is slow, we can just restrict to orthogonal. 
            weights = weights + h_1 @ X @ g.to(self.dtype) 
        return weights
            
class RightInvariantP(nn.Module):
    def __init__(self, group, dtype=torch.float):
        super().__init__()
        self.group = group
        self.dtype = dtype

    def forward(self, X):
        weights = torch.zeros_like(X).to(self.dtype) 

        for g in self.group.transformations:
            weights = weights + g.to(self.dtype) 
        return weights @ X

class LeftInvariantP(nn.Module):
    def __init__(self, group, dtype=torch.float):
        super().__init__()
        self.group = group
        self.dtype = dtype

    def forward(self, X):
        weights = torch.zeros_like(X).to(self.dtype) 

        for g in self.group.transformations:
            weights = weights + g.to(self.dtype) 
        return X @ weights


class LinearProjLayer(torch.nn.Module):
    # Implements sum_{g, h in G} h^{-1} * (W * g * x + b)

    def __init__(self, in_dim, out_dim, in_group, out_group=None, bias=True):
        super().__init__()

        if out_group is None:
            out_group = GroupRep.trivial(out_dim) # TODO: test this

        self.linear = nn.Linear(in_dim,out_dim, bias=bias)
        parametrize.register_parametrization(self.linear, "weight", EquivariantP(in_group, out_group))

        if bias:
            parametrize.register_parametrization(self.linear, "bias", RightInvariantP(out_group))
         
    def forward(self, x):
        return self.linear(x)

