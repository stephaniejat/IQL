import torch
import numpy as np
from typing import List

class GroupRep:
    def __init__(self, transformations: List[np.ndarray], dtype=torch.float):
        self.transformations = [torch.tensor(g, requires_grad = False, dtype=dtype) for g in transformations]
        # extract dimension of the domain
        self._dim = self.transformations[0].shape[1]

    @classmethod
    def trivial(cls, dim:int, length:int, dtype=torch.float):
        transformations = [torch.eye(dim, requires_grad = False, dtype=dtype)] * length
        return cls(transformations, dtype=dtype)

    def __getitem__(self, index):
        return self.transformations[index]

    def is_isomorphic(self, other):
        pass

    def check_closed(self):
        pass

    def to(self, device):
        self.transformations = [t.to(device) for t in self.transformations]

    def __len__(self):
        return len(self.transformations)
    
    @property
    def dim(self):
        return self._dim
