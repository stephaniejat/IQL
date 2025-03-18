import sys
import os
import numpy as np
import pandas as pd
import math
import argparse
import get_naive_reward as nr
from read_input_data import read_metadata, read_netlist, read_netlist_change_weight
from pprint import pprint


def rm_cluster_from_netlist(raw_nodes, net_list):
    # get cluster node id
    cluster_node_id = set([node.node_id for node in raw_nodes if node.node_type == 1])

    # remove cluster from net
    new_netlist = []
    for net in net_list:
        node_ids = list(set(net['node_ids']) - cluster_node_id)
        if len(node_ids) > 1:
            new_netlist.append(
                {
                    'net_id': net['net_id'],
                    'n_terminals': len(node_ids),
                    'node_ids': node_ids,
                    'weight': net['weight'],
                    'n_nets': net['n_nets'],
                }
            )
    return new_netlist

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input01_path', default=None, type=str, help='input 01_node_feature_list format file')
    parser.add_argument('--input02_path', default=None, type=str, help='input 02_metadata format file')
    parser.add_argument('--input04_path', default=None, type=str, help='input 04_netlist format file')
    parser.add_argument("--num_grids_on_canvas_shortside", default=32, type=int, help='num_grids_on_canvas_shortside')
    parser.add_argument("--place_cluster", default=True, type=bool, help='consider cluster or not')
    parser.add_argument("--wirelen_method", default='hpwl', type=str, help='wirelen_method for reward computation')
    parser.add_argument("--rudy_extend_ratio", default=[0.5], type=float, nargs='+', help='a list seperate by space')
    parser.add_argument("--penalty_factor", default=[0.1], type=float, nargs='+', help='a list seperate by space')
    parser.add_argument("--reset_anchor_weight", default=None, type=float, help='reset_anchor_weight')
    parser.add_argument("--reset_node_weight", default=None, type=float, help='reset_node_weight')
    parser.add_argument("--unittest_output_file", default=None, type=str, help='unittest_output_file')

    return parser


def read_raw_node_feature_list(path):
    """
    should not remove this, this process result is called by "write_output_data.py"
    """
    df = pd.read_csv(path)
    if 'no' not in df.columns:
        df.insert(loc=0, column='no', value=df.index)
    if 'x' not in df.columns or 'y' not in df.columns:
        df['x'] = df.apply(lambda row: row.lx + row.width / 2, axis=1)
        df['y'] = df.apply(lambda row: row.ly + row.height / 2, axis=1)
        df.drop(['lx'], axis=1, inplace=True)
        df.drop(['ly'], axis=1, inplace=True)
    return df 


class MockInputData:
    def __init__(
            self, input01_path, input02_path, input04_path, num_grids_on_canvas_shortside,
            reset_anchor_weight, reset_node_weight, updated_xy_vals):
        # load 01
        raw_nodes = read_raw_node_feature_list(input01_path)
        raw_nodes = self.update_xy_vals(raw_nodes, updated_xy_vals)
        raw_nodes = list(raw_nodes.itertuples(name='Node', index=False))
        # load 02
        self.metadata = read_metadata(input02_path)

        # compute grid length and board size
        self.grid_len = min(self.metadata['canvas_width'],
                            self.metadata['canvas_height']) / num_grids_on_canvas_shortside
        self.canvas_grid_number_width = math.ceil(self.metadata['canvas_width'] / self.grid_len)
        self.canvas_grid_number_height = math.ceil(self.metadata['canvas_height'] / self.grid_len)

        #  rescale x,y
        self.raw_nodes = self.node_feature_scale_preprocess(raw_nodes)
        self.nodes = [node for node in self.raw_nodes if node.node_type != 1]
        self.clster_nodes = [node for node in self.raw_nodes if node.node_type == 1]

        # check macros are placed
        assert -1 not in [node.x for node in self.nodes if node.node_type !=
                          1], "There is unplaced macro/anchor/ioport."
        assert -1 not in [node.y for node in self.nodes if node.node_type !=
                          1], "There is unplaced macro/anchor/ioport."

        # load 04
        if reset_anchor_weight is None and reset_node_weight is None:
            self.netlist = read_netlist(input04_path)
        else:
            self.netlist = read_netlist_change_weight(input04_path, self.nodes, reset_anchor_weight, reset_node_weight)
    
    def update_xy_vals(self, df, vals):
        if vals is None:
            return df
        else:
            vals = np.array(vals)
            df['x'] = vals[:, 0]
            df['y'] = vals[:, 1]
            return df

    def node_feature_scale_preprocess(self, nodes):
        if len(nodes) == 0:
            return nodes

        # Note: Update fields
        for idx, node in enumerate(nodes):
            node = node._replace(
                x=node.x / self.grid_len if node.x != -1 else -1,
                y=node.y / self.grid_len if node.y != -1 else -1,
                width=node.width / self.grid_len,
                height=node.height / self.grid_len,
                cluster_area=node.cluster_area / (self.grid_len * self.grid_len),
            )

            nodes[idx] = node

        return nodes


def compute_reward(
        input01_path, input02_path, input04_path, num_grids_on_canvas_shortside, place_cluster,
        wirelen_method, rudy_extend_ratio, penalty_factor, updated_xy_vals=None,
        reset_anchor_weight=None, reset_node_weight=None, unittest_output_file=None):

    input_data = MockInputData(
        input01_path, input02_path, input04_path, num_grids_on_canvas_shortside,
        reset_anchor_weight, reset_node_weight, updated_xy_vals)

    # prepare netlist and node_dict {node_id:Node} for computing reward
    if place_cluster:
        # check cluster nodes are placed
        assert -1 not in [node.x for node in input_data.clster_nodes if node.node_type ==
                          1], "There is unplaced cluster node."
        assert -1 not in [node.y for node in input_data.clster_nodes if node.node_type ==
                          1], "There is unplaced cluster node."

        # use the netlist contain clusters (raw netlist)
        netlist_for_reward = input_data.netlist
        placed_nodes = input_data.nodes + input_data.clster_nodes
    else:
        # use the netlist exclude clusters; consider nodes exclude clusters
        netlist_for_reward = rm_cluster_from_netlist(input_data.raw_nodes, input_data.netlist)
        placed_nodes = input_data.nodes

    placed_nodes_dict = {node.node_id: node for node in placed_nodes}

    # process rudy_extend_ratio
    extended_ratio_to_add = nr.get_extended_ratio_to_add(rudy_extend_ratio)

    # comput wirelength and congestion
    total_net_count, naive_wirelength, naive_congestion = nr.get_naive_reward(
        netlist_for_reward, placed_nodes_dict,
        input_data.canvas_grid_number_width, input_data.canvas_grid_number_height,
        extended_ratio_to_add, wirelen_method=wirelen_method, unittest_output_file=unittest_output_file
    )
    print('\n\033[95mResults\033[0m:')
    print(
        f'  \033[94mWirelength\033[0m: {round(naive_wirelength * input_data.grid_len, 4)} (grid scale: {round(naive_wirelength, 4)})')
    print(f'  \033[94mCongestion\033[0m: {round(naive_congestion, 4)}')

    # compute routability
    print('  \033[94mReward\033[0m:')
    for pf in penalty_factor:
        routability = nr.cal_routability(
            input_data, naive_wirelength * input_data.grid_len, naive_congestion, total_net_count,
            penalty_factor=pf, unittest_output_file=unittest_output_file)
        print(f'    pf ({pf}):\t{(-1)*round(routability, 4)}')
    return routability


def optim_reward(updated_xy_vals):
    reward = compute_reward(
        input01_path='c_wire-anchor_04-mixed-sided_9-1_o0_01_node_feature_list.csv', 
        input02_path = 'c_wire-anchor_04-mixed-sided_9-1_o0_02_metadata.json', 
        input04_path = 'c_wire-anchor_04-mixed-sided_9-1_o0_04_net_adjacency_list.csv', 
        num_grids_on_canvas_shortside = 42, 
        place_cluster = False,
        wirelen_method = 'hpwl', 
        rudy_extend_ratio = [0.5, 0.1], 
        penalty_factor = [0.2], 
        updated_xy_vals =updated_xy_vals,
        reset_anchor_weight=1.0, 
        reset_node_weight=None, 
        unittest_output_file=None
        )
    return reward

if __name__ == '__main__':
    '''
    Example command:
    python naive_reward_calculator.py \
        --input01_path c_wire-anchor_04-mixed-sided_9-1_o0_01_node_feature_list.csv \ # you can replace imput01 with your experiment output: 01_df_placed.csv
        --input02_path c_wire-anchor_04-mixed-sided_9-1_o0_02_metadata.json \
        --input04_path c_wire-anchor_04-mixed-sided_9-1_o0_04_net_adjacency_list.csv \
        --num_grids_on_canvas_shortside 20 \
        --reset_anchor_weight 1.0 \
        --rudy_extend_ratio 0.5 0.1 \
        --penalty_factor 0.1 0 1 10 \
        --place_cluster False \
        --wirelen_method hpwl \
        --unittest_output_file computation_detail.log 
    '''
    # load arguments
    args = vars(get_parser().parse_args())
    print('args:')
    pprint(args)
    if args['unittest_output_file'] is not None:
        f = open(args['unittest_output_file'], "w")
        args['unittest_output_file'] = f
        compute_reward(**args)
        f.close()
    else:
        compute_reward(**args)
