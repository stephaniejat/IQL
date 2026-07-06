import unittest
from unittest.mock import MagicMock
from gymnasium import spaces
from typing import List, Tuple

# from KRVI.sym_proj import LinearProjLayer
# from KRVI.group_rep import GroupRep
from KRVI.SymFeatureExtractor import SymmetricFeatureExtractor

class TestSymmetricFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Mocking GroupRep class
        self.mock_group_rep_10 = MagicMock()
        self.mock_group_rep_10.dim = 10
        self.mock_group_rep_20 = MagicMock()
        self.mock_group_rep_20.dim = 20
        self.mock_group_rep_30 = MagicMock()
        self.mock_group_rep_30.dim = 30

    def test_verify_layer_configs_correct(self):
        # Correct configuration
        layer_configs = [
            (10, 10, self.mock_group_rep_10, self.mock_group_rep_10),
            (10, 10, self.mock_group_rep_10, self.mock_group_rep_10)
        ]
        extractor = SymmetricFeatureExtractor
        try:
            extractor._verify_layer_configs(layer_configs)
        except AssertionError:
            self.fail("_verify_layer_configs raised AssertionError unexpectedly!")

    def test_verify_layer_configs_in_dim_mismatch(self):
        # Incorrect in_dim
        layer_configs = [
            (5, 20, self.mock_group_rep_10, self.mock_group_rep_20),
            (20, 10, self.mock_group_rep_20, self.mock_group_rep_30)
        ]
        extractor = SymmetricFeatureExtractor
        with self.assertRaises(AssertionError) as context:
            extractor._verify_layer_configs(layer_configs)
        self.assertIn("Dimaneison missmatch, group representation on space of dim 10, but given in_dim is 5", str(context.exception))

    def test_verify_layer_configs_out_dim_mismatch(self):
        # Incorrect out_dim
        layer_configs = [
            (10, 20, self.mock_group_rep_10, self.mock_group_rep_20),
            (20, 25, self.mock_group_rep_20, self.mock_group_rep_30)  
        ]
        extractor = SymmetricFeatureExtractor
        with self.assertRaises(AssertionError) as context:
            extractor._verify_layer_configs(layer_configs)
        self.assertIn("Dimaneison missmatch, group representation on space of dim 30, but given out_dim is 25", str(context.exception))

    def test_verify_in_out_dim_correct(self):
        # Correct configuration
        DIM_IN = 10
        out_features_dim = 30
        layer_configs = [
            (10, 20, self.mock_group_rep_10, self.mock_group_rep_20),
            (20, 30, self.mock_group_rep_20, self.mock_group_rep_30)
        ]
        try:
            SymmetricFeatureExtractor._verify_in_out_dim(DIM_IN, layer_configs, out_features_dim)
        except AssertionError:
            self.fail("_verify_in_out_dim raised AssertionError unexpectedly!")

    def test_verify_in_out_dim_first_layer_mismatch(self):
        # Incorrect first layer in_dim
        DIM_IN = 10
        out_features_dim = 30
        layer_configs = [
            (5, 20, self.mock_group_rep_10, self.mock_group_rep_20),  
            (20, 30, self.mock_group_rep_20, self.mock_group_rep_30)
        ]
        with self.assertRaises(AssertionError) as context:
            SymmetricFeatureExtractor._verify_in_out_dim(DIM_IN, layer_configs, out_features_dim)
        self.assertIn("Dimension missmatch, observation space with shape", str(context.exception))

    def test_verify_in_out_dim_last_layer_mismatch(self):
        # Incorrect last layer out_dim
        DIM_IN = 10
        out_features_dim = 25
        layer_configs = [
            (10, 20, self.mock_group_rep_10, self.mock_group_rep_20),
            (20, 30, self.mock_group_rep_20, self.mock_group_rep_30)  # out_dim is 30, but out_features_dim is 25
        ]
        with self.assertRaises(AssertionError) as context:
            SymmetricFeatureExtractor._verify_in_out_dim(DIM_IN, layer_configs, out_features_dim)
        self.assertIn("Dimension missmatch, feature space with shape", str(context.exception))

if __name__ == '__main__':
    unittest.main()