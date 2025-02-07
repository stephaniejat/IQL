import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.parametrize as parametrize
from KRVI.group_rep import GroupRep


class EquivariantP(nn.Module):
    def __init__(self, in_group, out_group, dtype=torch.float):
        super().__init__()
        self.in_group = [nn.Parameter(g) for g in in_group.transformations]
        self.out_group = [nn.Parameter(g) for g in out_group.transformations]
        for i,g in enumerate(self.in_group):
            self.register_parameter("g_{}".format(i), g)
        for i,g in enumerate(self.out_group):
            self.register_parameter("h_{}".format(i), g)
        self.dtype = dtype
        

    def forward(self, X):
        weights = torch.zeros_like(X).to(self.dtype) 

        for g,h in zip(self.in_group, self.out_group):
            h_1 = torch.inverse(h).to(self.dtype) #TODO: if this is slow, we can just restrict to orthogonal. 
            weights = weights + h_1 @ X @ g.to(self.dtype) 
        return weights
    
    # def to(self, device):
    #     # Move the module and its parameters to the specified device
    #     module = super().to(device)
    #     module.in_group = module.in_group.to(device)
    #     module.out_group = module.out_group.to(device)
    #     return module
            
class RightInvariantP(nn.Module):
    def __init__(self, group, dtype=torch.float):
        super().__init__()
        self.group = [nn.Parameter(g) for g in group.transformations]
        for i,g in enumerate(self.group):
            self.register_parameter("h_{}".format(i), g)
        self.dtype = dtype

    def forward(self, X):
        g = self.group[0]
        weights = torch.zeros_like(g).to(self.dtype) 

        for g in self.group:
            weights = weights + g.to(self.dtype) 
        return weights @ X

    # def to(self, device):
    #     # Move the module and its parameters to the specified device
    #     module = super().to(device)
    #     module.group = module.group.to(device)
    #     return module

class LeftInvariantP(nn.Module):
    def __init__(self, group, dtype=torch.float):
        super().__init__()
        self.group = [nn.Parameter(g) for g in group.transformations]
        for i,g in enumerate(self.group):
            self.register_parameter("g_{}".format(i), g)
        self.dtype = dtype

    def forward(self, X):
        g = self.group[0]
        weights = torch.zeros_like(g).to(self.dtype) 

        for g in self.group:
            weights = weights + g.to(self.dtype) 
        return X @ weights
    
    # def to(self, device):
    #     # Move the module and its parameters to the specified device
    #     module = super().to(device)
    #     module.group = module.group.to(device)
    #     return module


class LinearProjLayer(torch.nn.Module):
    # Implements sum_{g, h in G} h^{-1} * (W * g * x + b)

    def __init__(self, in_dim, out_dim, in_group, out_group=None, bias=True):
        super().__init__()

        if out_group is None:
            out_group = GroupRep.trivial(out_dim, len(in_group)) # TODO: test this

        self.linear = nn.Linear(in_dim,out_dim, bias=bias)
        parametrize.register_parametrization(self.linear, "weight", EquivariantP(in_group, out_group))

        if bias:
            parametrize.register_parametrization(self.linear, "bias", RightInvariantP(out_group))
         
    def forward(self, x):
        return self.linear(x)


class LinearProjLayer_inv(torch.nn.Module):
    # Implements  W * (sum_{g G} g )* x + b
    
    # TODO: This is much more numerically stable that passing out_group=None in the LinearProjLayer
    # I would like to understand why.
    def __init__(self, in_dim, out_dim, in_group, out_group=None, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_dim,out_dim, bias=bias)
        parametrize.register_parametrization(self.linear, "weight", LeftInvariantP(in_group))
         
    def forward(self, x):
        return self.linear(x)

