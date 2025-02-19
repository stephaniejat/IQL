import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import List, Tuple
import math

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from KRVI.sym_proj import LinearProjLayer, LeftInvariantP, LinearProjLayer_inv
from KRVI.group_rep import GroupRep

from utils import construct_90deg_block_rot_groups


class SymmetricFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) Required by the SB3 API
    :param layer_configs: (list of tuples) Each tuple contains (in_dim, out_dim, in_group, out_group)
    :param out_features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: spaces.Box, layer_configs: List[Tuple], out_features_dim:int):
        super().__init__(observation_space, out_features_dim)

        # Compute dim of flattened space 
        DIM_IN = math.prod(observation_space.shape)
        
        self._verify_in_out_dim(DIM_IN, layer_configs, out_features_dim)

        self._verify_layer_configs(layer_configs)

        # Create the layers based on the provided configurations
        layers = [nn.Flatten()]
        for in_dim, out_dim, in_group, out_group in layer_configs:
            layers.append(LinearProjLayer(in_dim, out_dim, in_group, out_group, bias=True))
            layers.append(nn.ReLU())

        self.extractor = nn.Sequential(*layers)
    
    @staticmethod
    def _verify_in_out_dim(DIM_IN, layer_configs, out_features_dim):
        # Verify first layer correct
        first_layer_conf = layer_configs[0]
        first_layer_in_dim = first_layer_conf[0]
        assert DIM_IN == first_layer_in_dim, "Dimension missmatch, observation space with shape {} and first layer {}".format(
            DIM_IN,
            first_layer_conf
        )

        last_layer_conf = layer_configs[-1]
        final_out_dim = last_layer_conf[1]
        assert out_features_dim == final_out_dim, "Dimension missmatch, feature space with shape {} and last layer {}".format(
            out_features_dim,
            last_layer_conf
        )

    @staticmethod
    def _verify_layer_configs(layer_configs: List[Tuple]):
        for config in layer_configs:
            in_dim, out_dim, in_group, out_group = config
            assert in_dim == in_group.dim, "Dimaneison missmatch, group representation on space of dim {}, but given in_dim is {}".format(
                in_group.dim,
                in_dim
            )
            assert out_dim == out_group.dim, "Dimaneison missmatch, group representation on space of dim {}, but given out_dim is {}".format(
                out_group.dim,
                out_dim
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(observations)

class _SymmetricFeatureExtractor(BaseFeaturesExtractor):
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
