import torch
import numpy as np
from typing import List

class GroupRep:
    def __init__(self, transformations: List[np.ndarray], dtype=torch.float):
        self.transformations = [torch.tensor(g, requires_grad = False, dtype=dtype) for g in transformations]

    @classmethod
    def trivial(cls, dim:int, dtype=torch.float):
        transformations = [torch.eye(dim, requires_grad = False, dtype=dtype)]

    def __getitem__(self, index):
        return self.transformations[index]

    def is_isomorphic(self, other):
        pass

    def check_closed(self):
        pass
