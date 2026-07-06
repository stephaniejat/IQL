import unittest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from KRVI.sym_proj import LinearProjLayer
from utils import construct_90deg_block_rot_groups
from KRVI.group_rep import GroupRep
from KRVI.sym_proj import EquivariantP, RightInvariantP


class TestEquivariantProjLayer(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)

    def test_layer_equivariance(self):
        DIM_IN = 12
        DIM_OUT = 2
        G12, G2 = construct_90deg_block_rot_groups(DIM_IN)

        # Verify Layer is equivariant
        gell = LinearProjLayer(DIM_IN, DIM_OUT, G12, G2, bias=True)
        x = torch.randn(DIM_IN, dtype=torch.float)
        for index in range(4):
            self.assertTrue(torch.allclose(G2[index] @ gell(x), gell(G12[index] @ x), atol=1e-6))

if __name__ == '__main__':
    unittest.main()