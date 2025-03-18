import json
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn import preprocessing
import os
import copy
import math
from itertools import combinations
from ast import literal_eval as make_tuple
# from tools.helper.normalization import normalization, sklearn_preprocessing
# from tools.helper.node_sorting import sort_by_size_linkcount, sort_by_linkcount
# from tools.helper.node_sorting import sort_by_link2anchor, sort_by_anchorSizeNlink
from typing import List, Optional, Dict
import copy


def read_raw_node_feature_list(path):
    """
    should not remove this, this process result is called by "write_output_data.py"
    """
    df = pd.read_csv(path)
    _check_01_column_name(df, path)
    df.insert(loc=0, column='no', value=df.index)
    df['x'] = df.apply(lambda row: row.lx + row.width / 2, axis=1)
    df['y'] = df.apply(lambda row: row.ly + row.height / 2, axis=1)
    df.drop(['lx'], axis=1, inplace=True)
    df.drop(['ly'], axis=1, inplace=True)
    return list(df.itertuples(name='Node', index=False))


def _check_01_column_name(df_01, node_feature_list_path):
    predefined_columns = [
        'node_id', 'node_type', 'width', 'height', 'lx', 'ly', 'pin_count',
        'cluster_area', 'internal_net', 'group_id', 'region_id'
    ]
    assert list(df_01.columns) == predefined_columns, \
        f"\nThe columns in {node_feature_list_path} \
        \n is mismatch to wiki page: https://wiki.mediatek.inc/pages/viewpage.action?pageId=788284357"


def read_ans_node_feature_list(List_Node):
    """
    should not remove this, this process result is called by "write_output_data.py"
    """
    df = pd.DataFrame(List_Node)
    df.insert(loc=df.shape[-1], column='action', value=-1)  # add action columne
    return list(df.itertuples(name='Node', index=False))


def read_node_feature_list(List_Node):
    """
    load 01_node_feature_list.csv into pd.DataFrame();
    do further process to form list(Node())
    """
    df = pd.DataFrame(List_Node)

    # Note: will assign neighbor_count value later when connections is ready
    df['neighbor_count'] = 0

    # Note: sort by total_connection_count_of_superior_nodes later
    df['total_connection_count_of_superior_nodes'] = 0

    # Note: for output data in the future
    df['place_order'] = None
    df['wirelength'] = None
    df['congestion'] = None
    df['reward'] = None

    print('[done] read_node_feature')
    return list(df.itertuples(name='Node', index=False))


def read_metadata(metadata_path):
    """
    load 02_metadata.json to dict();
    the order of element in dict() is the same as metadata.json
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f, object_pairs_hook=OrderedDict)

    print('[done] read_metadata')
    return metadata


def read_connections(connections_path, nodes, node_indices):
    """
    transform 03_weighted_adjacency_matrix_sparse.csv to adjacency matrix (2D np.array())
    """
    # Note: initialize connections
    n_nodes = len(nodes)
    matrix = np.zeros((n_nodes, n_nodes))

    # Note: fill info to Connetion_Count_Matrix
    fp = open(connections_path, "r")
    for src_node, line in enumerate(fp):
        # skip blank line (blank line means the node is isolated from other nodes)
        if not line.strip():
            continue
        line = make_tuple(line.strip())
        # Note: if line contain only one item:
        #         (1,3) --make_tuple()---> (1,3)
        #       however, we expect $line should formed as:
        #         (1,3),(2,1),(3,0) --make_tuple()---> ((1,3),(2,1),(3,0))
        #       cases like (1,3) need further transform to tuple(tuple())
        if not isinstance(line[0], tuple):
            line = [line]

        for (dst_node, link_weight) in line:
            matrix[src_node][node_indices[dst_node]] = link_weight
    fp.close()

    print('[done] read_connections')
    return matrix


def read_netlist(_netlist_path):
    """
    load 04_netlist to list()
    """
    df = pd.read_table(_netlist_path, header=None)
    n_dims = len(df)
    netlist = [[]] * n_dims
    for net_index, net in enumerate(df.values.tolist()):
        net_id = net_index + 1

        # Warning for switching to the new 03,04 files. (Deprecat once all data are re-generated.)
        assert len(make_tuple(net[0])) == 4, 'Your 04_net_adjacency_list.csv may be old version. Please check again.'
        n_terminals, n_nets, weight, node_ids = make_tuple(net[0])
        netlist[net_index] = {'net_id': net_id,
                              'n_nets': n_nets,
                              'weight': weight,
                              'node_ids': node_ids,
                              'n_terminals': n_terminals}

    print('[done] read_netlist')
    return netlist


def read_netlist_change_weight(_netlist_path, _nodes, anchor_weight, node_weight):
    anchor_node_id = set([node.node_id for node in _nodes if node.node_type == 4])

    df = pd.read_table(_netlist_path, header=None)
    n_dims = len(df)
    netlist = [[]] * n_dims
    for net_index, net in enumerate(df.values.tolist()):
        net_id = net_index + 1

        # Warning for switching to the new 03,04 files. (Deprecat once all data are re-generated.)
        assert len(make_tuple(net[0])) == 4, 'Your 04_net_adjacency_list.csv may be old version. Please check again.'
        n_terminals, n_nets, weight, node_ids = make_tuple(net[0])
        if anchor_node_id.intersection(set(node_ids)):
            netlist[net_index] = {'net_id': net_id,
                                  'n_nets': n_nets,
                                  'weight': weight if anchor_weight is None else anchor_weight,
                                  'node_ids': node_ids,
                                  'n_terminals': n_terminals}
        else:
            netlist[net_index] = {'net_id': net_id,
                                  'n_nets': n_nets,
                                  'weight': weight if node_weight is None else node_weight,
                                  'node_ids': node_ids,
                                  'n_terminals': n_terminals}

    print('[done] read_netlist')
    return netlist


def gen_weighted_connection(_netlist, _nodes):
    nodeID2Index = {node.node_id: node.no for node in _nodes}
    conn = np.zeros((len(_nodes), len(_nodes)))
    for net in _netlist:
        combs = combinations(net['node_ids'], 2)
        for (node_1, node_2) in combs:
            conn[nodeID2Index[node_1], nodeID2Index[node_2]] += (net['weight'] * net['n_nets'])
            conn[nodeID2Index[node_2], nodeID2Index[node_1]] += (net['weight'] * net['n_nets'])
    non_zero_element = len(np.where(conn)[0])

    # fill in diagonal value
    non_zero_diagonal = 0
    for node in _nodes:
        if node.internal_net != 0:
            non_zero_diagonal += 1
            conn[nodeID2Index[node.node_id], nodeID2Index[node.node_id]] = node.internal_net
    print(f'* non-zero element (shareEdge): {non_zero_element} ({non_zero_element/2})')
    print('* non-zero diagonal:', non_zero_diagonal)
    print(f'* Total gnn edge (shareEdge): {non_zero_element+non_zero_diagonal} ({non_zero_element/2 + non_zero_diagonal})')
    return conn


def read_region(region_path, id_start_from):
    """
    load 06 to the same format as node.blockages
    """
    df = pd.read_csv(region_path)
    assert len(df) >= 1, '06_region should at least contain "canvas region".'

    df['width'] = df.apply(lambda region: abs(region.lx - region.ux), axis=1)
    df['height'] = df.apply(lambda region: abs(region.ly - region.uy), axis=1)
    df['x'] = df.apply(lambda region: (region.lx + region.ux) / 2, axis=1)
    df['y'] = df.apply(lambda region: (region.ly + region.uy) / 2, axis=1)
    df['node_id'] = [id_start_from + i for i in range(len(df))]

    # Blockage
    if len(df[df['mask_type'] == 0]) > 0:
        blockages_df = df[df['mask_type'] == 0].copy()
        blockages_df['node_type'] = 3
        blockages = list(blockages_df.itertuples(name='Node', index=False))
    else:
        blockages = []

    # Power domain
    if len(df[df['mask_type'] == 1]) > 0:
        power_domain_df = df[df['mask_type'] == 1].copy()
        power_domain_df['node_type'] = 5
        power_domain = list(power_domain_df.itertuples(name='Node', index=False))
    else:
        power_domain = []

    print('[done] read_region')
    return blockages, power_domain


def node_feature_normalization(nodes_list, node_feature_columns, node_feature_std_func):
    """
    do standardization
    """
    nodes_df = pd.DataFrame(nodes_list)[node_feature_columns]

    assert node_feature_std_func, f"node_feature_std_func: '{node_feature_std_func}' is not defined."
    nodes = normalization(
        nodes_df,
        sklearn_preprocessing(preprocessing.StandardScaler()),
        do_vertically=True
    )
    nodes_df = pd.DataFrame(nodes, index=nodes_df.index,
                            columns=nodes_df.columns)
    return list(nodes_df.itertuples(name='Node', index=False))


def metadata_normalization(metadata, meta_std_func):
    """
    do standardization
    """
    meta_df = pd.DataFrame.from_dict(metadata, orient='index').T

    assert meta_std_func, f"meta_std_func: '{meta_std_func}' is not defined."
    meta_matrix = normalization(
        meta_df,
        sklearn_preprocessing(preprocessing.StandardScaler()),
        do_vertically=False
    )

    return meta_matrix


def get_meta_matrix(metadata):
    """
    turn metadata to np.array
    """
    meta_matrix = [metadata_value for metadata_value in metadata.values()]
    return np.array(meta_matrix)


def get_data_report(nodes, connections, note=''):

    # node type
    TYPE_MACRO = 0
    TYPE_CLUSTER = 1
    TYPE_IO_PORT = 2
    TYPE_BLOCKAGE = 3
    TYPE_ANCHOR = 4

    try:
        df_node = pd.DataFrame(nodes)

        idx_macro = df_node[df_node['node_type'] == TYPE_MACRO]['no'].values
        idx_cluster = df_node[df_node['node_type'] == TYPE_CLUSTER]['no'].values
        idx_ioport = df_node[df_node['node_type'] == TYPE_IO_PORT]['no'].values
        idx_blockage = df_node[df_node['node_type'] == TYPE_BLOCKAGE]['no'].values
        idx_anchor = df_node[df_node['node_type'] == TYPE_ANCHOR]['no'].values

        num_macro = len(idx_macro)
        num_cluster = len(idx_cluster)
        num_ioport = len(idx_ioport)
        num_blockage = len(idx_blockage)
        num_anchor = len(idx_anchor)
        num_total = len(df_node)

        title_list = ['macro', 'cluster', 'io_port', 'blockage', 'anchor']
        idx_list = [idx_macro, idx_cluster, idx_ioport, idx_blockage, idx_anchor]
        sum_conn = np.zeros((len(idx_list), len(idx_list)), dtype=int)
        count_conn = np.zeros((len(idx_list), len(idx_list)), dtype=int)
        max_conn = np.zeros((len(idx_list), len(idx_list)), dtype=int)
        for i, idx_i in enumerate(idx_list):
            for j, idx_j in enumerate(idx_list):
                conn = connections[idx_i, :][:, idx_j]
                sum_conn[i][j] = np.sum(conn)
                count_conn[i][j] = np.sum(conn > 0)
                max_conn[i][j] = np.max(conn, initial=0)

        report = (
            '-' * 50 + '\n'
            f'# user note: {note}\n\n'
            f'## node type statistic\n'
            f'- num_macro: {num_macro}\n- num_cluster: {num_cluster}\n'
            f'- num_ioport: {num_ioport}\n- num_blockage: {num_blockage}\n'
            f'- num_anchor: {num_anchor}\n- num_total: {num_total}\n\n'
            f'## connection relation\n(order: {title_list})\n\n'
            f'- count:\n{count_conn}\n(sum: {np.sum(count_conn)})\n\n'
            f'- sum:\n{sum_conn}\n(sum: {np.sum(sum_conn)})\n\n'
            f'- max:\n{max_conn}\n(sum: {np.sum(max_conn)})\n'
            + '-' * 50
        )
    except:
        report = 'error'

    return report


class InputData:
    def __init__(
        self, input_data_path,
        node_feature_std, node_feature_std_func,
        meta_std, node_feature_columns, meta_std_func,
        num_grids_on_canvas_shortside,
        remove_cluster, reset_macro_design,
        sorting_method: str,
        reset_anchor_weight: Optional[float],
        reset_node_weight: Optional[float],
        n_terminal_thr: Optional[int] = None,
    ):
        # set basic config (settings)
        self.base_input_data_path = '/'.join(input_data_path.split('/')[:-1])

        # Note 1: read 01 to 04 files
        # 1-1. define path to data
        node_feature_list_path = input_data_path + '01_node_feature_list.csv'
        metadata_path = input_data_path + '02_metadata.json'
        connections_path = input_data_path + '03_weighted_adjacency_matrix_sparse.csv'
        netlist_path = input_data_path + '04_net_adjacency_list.csv'
        region_path = input_data_path + '06_region.csv'

        # 1-2. load files to following form
        """
        self.nodes -> list(Node())
        self.metadata -> dict()
        self.meta_matrix -> np.array((1, n_meta_feature))
        self.connections -> np.array((n_nodes, n_nodes))
        self.netlist -> dict()
        """
        self.raw_nodes = read_raw_node_feature_list(node_feature_list_path)
        self.nodes = read_node_feature_list(self.raw_nodes.copy())
        self.node_indices = {node.node_id: node.no for node in self.nodes}
        self.metadata = read_metadata(metadata_path)

        if reset_anchor_weight is None and reset_node_weight is None:
            self.connections = read_connections(connections_path, self.nodes, self.node_indices)
            self.netlist = read_netlist(netlist_path)
        else:
            self.netlist = read_netlist_change_weight(
                netlist_path, self.nodes, reset_anchor_weight, reset_node_weight)
            self.connections = gen_weighted_connection(self.netlist, self.nodes)

        # prepare connection for gnn when
        if n_terminal_thr is not None:
            net_terminal_info = self.netlist.copy()
            net_terminal_info = [net for net in net_terminal_info if net['n_terminals'] < n_terminal_thr]

            # convert net to two-pin edge
            self.gnn_connections = gen_weighted_connection(
                net_terminal_info, self.nodes)
        else:
            self.gnn_connections = copy.deepcopy(self.connections)

        assert len(list(filter(lambda node: node.node_type == 3, self.nodes))) == 0,\
            '\n Blockages exist in both 06 and 01 files. Please check!'
        self.blockages, self.power_domain = read_region(region_path, len(self.nodes))

        # original data's report
        self.data_report_raw = get_data_report(nodes=self.nodes, connections=self.connections,
                                               note='(the data before process)')
        print(self.data_report_raw)

        # calculate grid_len to decide number of grid to split
        self.grid_len = self.get_grid_len(
            min(self.metadata['canvas_width'], self.metadata['canvas_height']),
            num_grids_on_canvas_shortside
        )

        # canvas padding
        self.canvas_grid_number_width, self.canvas_grid_number_height, padd_blockage = self.gen_padding_blockage(
            self.metadata['canvas_width'],
            self.metadata['canvas_height'],
            self.grid_len,
            id_start_from=len(self.blockages) + len(self.nodes)
        )
        if len(padd_blockage) != 0:
            self.blockages += padd_blockage

        # Note 2: setting/saving needed configs
        self.remove_cluster = remove_cluster
        self.reset_macro_design = reset_macro_design

        # Note 3: node feature preprocess and placing order sorting
        self.nodes = self.node_feature_scale_preprocess(self.nodes)
        self.blockages = self.node_feature_scale_preprocess(self.blockages)
        self.power_domain = self.node_feature_scale_preprocess(self.power_domain)
        self.nodes = self.node_sorting(self.nodes, self.connections, self.node_indices, sorting_method)

        # Note 4: whether to add cluster nodes as input to placement model
        if self.remove_cluster:
            self.nodes, self.cluster_nodes = self.remove_cluster_data(self.nodes)
            self.metadata['cluster_count'] = 0

        # retrieve io-port and blockage
        self.IOports = self.get_ioport_data(self.nodes)

        # Note 5. do normalization if required
        if node_feature_std:
            self.nodes = node_feature_normalization(
                self.nodes, node_feature_columns, node_feature_std_func)

        if meta_std:
            self.meta_matrix = metadata_normalization(
                self.metadata, meta_std_func)
            # print('meta_std:', meta_std)
            # print("self.meta_matrix shape:", self.meta_matrix.shape)

        else:
            self.meta_matrix = get_meta_matrix(self.metadata).reshape(1, len(self.metadata))
            # print("self.meta_matrix shape:", self.meta_matrix.shape)

        # Note 6. optimal action
        ans_nodes_ = read_ans_node_feature_list(self.raw_nodes.copy())
        ans_nodes = self.ans_node_position_to_action(ans_nodes_)
        self.optimal_actions = self.get_optimal_action(self.nodes, ans_nodes)

        # using data's report
        self.data_report_used = get_data_report(nodes=self.nodes, connections=self.connections,
                                                note='experiment used data (the data after process)')
        print(self.data_report_used)

    def node_feature_scale_preprocess(self, nodes):
        if len(nodes) == 0:
            return nodes

        # Note: Update fields
        for idx, node in enumerate(nodes):
            if node.node_type != 3 and node.node_type != 5:
                # reset macro's (x, y) to (-1, -1)
                if self.reset_macro_design and node.node_type == 0:
                    node = node._replace(x=-1, y=-1)

                node = node._replace(
                    x=node.x / self.grid_len if node.x != -1 else -1,
                    y=node.y / self.grid_len if node.y != -1 else -1,
                    width=node.width / self.grid_len,
                    height=node.height / self.grid_len,
                    cluster_area=node.cluster_area / (self.grid_len * self.grid_len),
                )

            else:
                node = node._replace(
                    x=node.x / self.grid_len,
                    y=node.y / self.grid_len,
                    width=node.width / self.grid_len,
                    height=node.height / self.grid_len,
                    lx=node.lx / self.grid_len,
                    ly=node.ly / self.grid_len,
                    ux=node.ux / self.grid_len,
                    uy=node.uy / self.grid_len,
                )

            nodes[idx] = node

        return nodes

    def node_sorting(
        self, nodes: List, connections: np.ndarray, node_indices: Dict, sorting_method: str = 'default'
    ):
        """
        : sorting_method: str
            1) 'default': prioritize placement order according cluster_area (big->small) first, then link count
            2) 'max_link_count_first': prioritize placement order according to link count with placed nodes
            3) 'link_to_anchor_first': place node link to anchor first
            4) 'anchor_size_nlink': placed order: macro has link with anchor -> cluster area -> pin count
                the placed order follows PD design rule
        """
        # seperate nodes that needed sorting and not needed sorting
        nodes_tobe_sorted, nodes_not_sort = [], []
        for node in nodes:
            if (node.node_type == 0) or (node.node_type == 4):
                nodes_tobe_sorted.append(node)
            else:
                nodes_not_sort.append(node)

        if sorting_method == 'default':
            sorted_nodes = sort_by_size_linkcount(nodes_tobe_sorted, connections, node_indices)

        elif sorting_method == 'max_link_count_first':
            sorted_nodes = sort_by_linkcount(nodes_tobe_sorted, connections, node_indices)

        elif sorting_method == 'link_to_anchor_first':
            sorted_nodes = sort_by_link2anchor(nodes_tobe_sorted, connections, node_indices)

        elif sorting_method == 'anchor_size_nlink':
            sorted_nodes = sort_by_anchorSizeNlink(nodes_tobe_sorted, connections, node_indices)

        return sorted_nodes + nodes_not_sort

    def remove_cluster_data(self, nodes):  # , connections):
        """
        Get $nodes that remove cluster nodes.
        node_type:
            0 = macro
            1 = cluster
            2 = I/O port
            3 = blockage (placing-forbidden-area)
            4 = anchor
        """
        # Note: Process on $nodes
        # get nodes list exclude cluster nodes
        nodes__exclude_cluster = list(
            filter(lambda node: node.node_type != 1, nodes))
        cluster_nodes = list(
            filter(lambda node: node.node_type == 1, nodes))
        return nodes__exclude_cluster, cluster_nodes

    def get_ioport_data(self, nodes):
        ioport_nodes = list(
            filter(lambda node: node.node_type == 2, nodes))
        return ioport_nodes

    def get_canvas_grid_number_width(self):
        return self.canvas_grid_number_width

    def get_canvas_grid_number_height(self):
        return self.canvas_grid_number_height

    def get_base_input_data_path(self):
        return self.base_input_data_path

    def get_grid_len(self, physical_shortside_length, num_grids_on_canvas_shortside):
        """
        calcualte grid_len by deviding canvas_short_side_length by `num_grids_on_canvas_shortside`
        """
        if num_grids_on_canvas_shortside is None:
            # TODO: (YW) may apply yi-chen's calculat_grid_wastes in the future
            num_grids_on_canvas_shortside = 32

        grid_len = physical_shortside_length / num_grids_on_canvas_shortside
        return grid_len

    def gen_padding_blockage(self, canvas_w, canvas_h, grid_len, id_start_from):
        # Note: if canvas_w >= canvas_h, padding on right; o.w. padding one the bottom of canvas.
        canvas_grid_number_width = math.ceil(canvas_w / grid_len)
        canvas_grid_number_height = math.ceil(canvas_h / grid_len)

        if canvas_w % grid_len == 0:
            # both sides are divisible by `grid_len`
            if canvas_h % grid_len == 0:
                return canvas_grid_number_width, canvas_grid_number_height, []

            # only `canvas_w` side is divisible by `grid_len`
            else:
                lx = 0
                ly = canvas_h
                ux = canvas_w
                uy = canvas_grid_number_height * grid_len

        # only `canvas_h` side is divisible by `grid_len`
        elif canvas_w % grid_len != 0:
            lx = canvas_w
            ly = 0
            ux = canvas_grid_number_width * grid_len
            uy = canvas_h

        if abs(ux - lx) == 0 or abs(uy - ly) == 0:
            return canvas_grid_number_width, canvas_grid_number_height, []

        padding_blockages = pd.DataFrame(
            [[0, 3, lx, ly, ux, uy, -1,
              abs(ux - lx), abs(uy - ly), (ux + lx) / 2, (uy + ly) / 2, id_start_from]],
            columns=[
                'mask_type', 'node_type', 'lx', 'ly', 'ux', 'uy', 'region_id',
                'width', 'height', 'x', 'y', 'node_id']
        )
        return canvas_grid_number_width, canvas_grid_number_height, \
            list(padding_blockages.itertuples(name='Node', index=False))

    def ans_node_position_to_action(self, ans_nodes):
        """
        from real, center position to action

        Note: 
        1. this function only support integer (grid_x, grid_y) now
        2. the ans action is only for reference (in most cases, we don't have answer)
        """
        # Note: Update fields
        for idx, node in enumerate(ans_nodes):
            # only update macro & non -1 nodes
            if node.node_type == 0 and node.x != -1 and node.x >= 0:
                grid_x = node.x / self.grid_len - 0.5
                grid_y = node.y / self.grid_len - 0.5

                # check both (grid_x, grid_y) are integer, or save action: -1
                # here using abs_diff to check for avoid float format problem
                grid_x_valid = round(grid_x, 0)
                grid_y_valid = round(grid_y, 0)
                if abs(grid_x_valid - grid_x) <= 1e-4 and abs(grid_y_valid - grid_y) <= 1e-4:
                    action = int(self.canvas_grid_number_width * grid_y_valid + grid_x_valid)
                else:
                    action = -1

                node = node._replace(action=action)
            ans_nodes[idx] = copy.deepcopy(node)
        return ans_nodes

    def get_optimal_action(self, nodes, ans_nodes):
        optimal_actions = []
        for node in nodes:
            if node.node_type == 0:  # only save macro's action
                no_ = int(node.no)
                if no_ < len(ans_nodes):
                    opt_action = ans_nodes[no_].action
                    optimal_actions.append(opt_action)
        return optimal_actions.copy()
