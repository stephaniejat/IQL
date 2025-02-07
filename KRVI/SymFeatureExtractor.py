import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from KRVI.sym_proj import LinearProjLayer, LeftInvariantP, LinearProjLayer_inv
from KRVI.group_rep import GroupRep

from utils import construct_90deg_block_rot_groups


class SymmetricFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, in_group=None, out_group=None, out_features_dim: int = 4):
        super().__init__(observation_space, out_features_dim)

        # Compute dim of flattened space
        DIM_IN = observation_space.shape[0] * observation_space.shape[1]
        G14, G2 = construct_90deg_block_rot_groups(14)
        G12, G2 = construct_90deg_block_rot_groups(12)
        G10, G2 = construct_90deg_block_rot_groups(10)
        G8, G2 = construct_90deg_block_rot_groups(8)
        G6, G2 = construct_90deg_block_rot_groups(6)
        G4, G2 = construct_90deg_block_rot_groups(4)

        # self.equivariant_mlp = LinearProjLayer(DIM_IN, DIM_OUT, in_group, out_group, bias = True)
        self.extractor = nn.Sequential(
            nn.Flatten(),
            # LinearProjLayer(14, 12, G14, G12, bias = True),
            # nn.ReLU(),
            # LinearProjLayer(12, 10, G12, G10, bias = True),
            # nn.ReLU(),
            # LinearProjLayer(10, 8, G10, G8, bias = True),
            # nn.ReLU(),
            # LinearProjLayer(8, 6, G8, G6, bias = True),
            # nn.ReLU(),
            LinearProjLayer(14, 4, G14, out_group=None, bias = True),
            nn.ReLU()
        )
        

        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(observations)
