import numpy as np
import math
from itertools import combinations
from tabulate import tabulate


def get_extended_ratio_to_add(rudy_extend_ratio):
    # Process rudy_extend_ratio for later use
    # [1] represents the original region
    if len(rudy_extend_ratio) > 0:
        rudy_extend_ratio_array = np.array([1] + rudy_extend_ratio)
        # trick: add the values ​​proportionally.
        # For example, if rudy_extend_ratio = [1, 0.5, 0.2],
        # then we would like to have new list [(1 - 0.5), (0.5 - 0.2), 0.2]
        # We first fill "(1 - 0.5) * rudy" into center region;
        #  then, fill "(0.5 - 0.2) * rudy" into center region and the region that extend by one grid;
        #  finally, fill "0.2 * rudy" into above regions and the region that extend by two grid;
        extended_ratio_to_add = list(rudy_extend_ratio_array[:-1] -
                                     rudy_extend_ratio_array[1:]) + rudy_extend_ratio[-1:]
        for ratio in extended_ratio_to_add:
            assert ratio <= 1, 'all value in rudy_extend_ratio ({self.rudy_extend_ratio}) should fall in (0,1].'
            assert ratio >= 0, 'rudy_extend_ratio ({self.rudy_extend_ratio}) should be a decreasing sequence.'
    else:
        extended_ratio_to_add = [1]

    return extended_ratio_to_add


class Net:
    def __init__(self, net_dict, unittest_output_file):
        super(Net, self).__init__()
        self.node_ids = net_dict['node_ids']
        self.n_nets = net_dict['n_nets']
        self.weight = net_dict['weight']
        self.unittest_output_file = unittest_output_file

    def get_net_reward(
        self, placed_nodes_dict, canvas_grid_number_width, canvas_grid_number_height, extended_ratio_to_add,
        wirelen_method
    ):
        Xs, Ys = zip(*[[placed_nodes_dict[node_id].x, placed_nodes_dict[node_id].y] for node_id in self.node_ids])
        node_types = [placed_nodes_dict[node_id].node_type for node_id in self.node_ids]

        # make sure all nodes have been placed (i.e both x and y not equal to -1)
        assert -1 not in Xs, \
            f"Node (id: { self.node_ids[Xs.index(-1)]}) expected being placed but got x = -1."
        assert -1 not in Ys, \
            f"Node (id: { self.node_ids[Ys.index(-1)]}) expected being placed but got y = -1."

        # compute "coordinates" boundary
        left, right, bottom, top = min(Xs), max(Xs), min(Ys), max(Ys)

        # compute "matrix" boundary
        matrix_left, matrix_right, matrix_bottom, matrix_top, special_case = self._get_matrix_boundary(
            left, right, bottom, top, canvas_grid_number_width, canvas_grid_number_height)

        # derive wirelength  of net
        if wirelen_method == 'hpwl':
            net_wirelength, net_count = self._get_net_hpwl(left, right, bottom, top)
            net_wirelength_for_congestion = net_wirelength

        elif wirelen_method == 'manhattan':
            net_wirelength, net_count, two_pin_dist = self._get_net_manhattan_distance(placed_nodes_dict)
            net_wirelength_for_congestion = abs(matrix_right - matrix_left) + abs(matrix_top - matrix_bottom)

        # derive congestion map of net
        if 4 in node_types:
            net_congestion_map = np.zeros((canvas_grid_number_height, canvas_grid_number_width))

        else:
            net_congestion_map, rudy_values, net_area = self._get_net_congestion(
                matrix_left, matrix_right, matrix_bottom, matrix_top, net_wirelength_for_congestion,
                canvas_grid_number_width, canvas_grid_number_height, extended_ratio_to_add, wirelen_method, special_case)

        # For unit test results check
        if self.unittest_output_file is not None:
            if wirelen_method == 'manhattan':
                print(f'net HPWL ({wirelen_method}): {net_wirelength} <--{two_pin_dist} (W_wl: {self.n_nets*self.weight})',
                      file=self.unittest_output_file)
            else:
                print(f'net HPWL ({wirelen_method}): {net_wirelength} (W_wl: {self.n_nets*self.weight})',
                      file=self.unittest_output_file)

            if 4 not in node_types:
                print('net RUDY (HPWL/AREA): ', file=self.unittest_output_file)
                print(
                    f'  |- Boundary (L,R,B,T): ({matrix_left}, {matrix_right}, {matrix_bottom}, {matrix_top})', file=self.unittest_output_file)
                print(f'  |- HPWL: {np.round(net_wirelength_for_congestion, 3)}', file=self.unittest_output_file)
                print(f'  |- AREA: {np.round(net_area, 3)}', file=self.unittest_output_file)
                print(f'  |- HPWL/AREA: {np.round(net_wirelength_for_congestion/net_area, 3)} (W_rudy: {self.n_nets})',
                    file=self.unittest_output_file)
                print(f'  |- Extd Rudy: {np.round(rudy_values, 3)}\n', file=self.unittest_output_file)

                table = tabulate(net_congestion_map, headers=list(
                    range(net_congestion_map.shape[1])), showindex="always", floatfmt=".3f")
                print(table, file=self.unittest_output_file)
            else:
                print('net RUDY (HPWL/AREA): skip virtual net ', file=self.unittest_output_file)

        return net_count, net_wirelength, net_congestion_map

    def _get_net_hpwl(
        self, coordinates_left_bound, coordinates_right_bound, coordinates_lower_bound, coordinates_upper_bound
    ):
        """
        get the boundary of the net and then compute hpwl
        """
        net_hpwl = abs(coordinates_right_bound - coordinates_left_bound) + \
            abs(coordinates_upper_bound - coordinates_lower_bound)
        return net_hpwl, self.n_nets

    def _get_net_manhattan_distance(self, placed_nodes_dict):
        """
        get all combination of two nodes and compute the mahattan distance of the two nodes
        """
        two_pin_dist = []
        manhattan_dist = 0
        paired_nodes = list(combinations(self.node_ids, 2))
        two_pin_net_count = len(paired_nodes)
        for id_i, id_j in paired_nodes:
            width = abs(placed_nodes_dict[id_i].x - placed_nodes_dict[id_j].x)
            height = abs(placed_nodes_dict[id_i].y - placed_nodes_dict[id_j].y)
            manhattan_dist += (width + height)
            two_pin_dist.append((width + height))

        return manhattan_dist, two_pin_net_count * self.n_nets, two_pin_dist

    def _get_net_congestion(
        self, matrix_left, matrix_right, matrix_bottom, matrix_top, net_hpwl,
        canvas_grid_number_width, canvas_grid_number_height, extended_ratio_to_add, wirelen_method, special_case
    ):
        # 1. init net_congestion_map
        net_congestion_map = np.zeros(
            (canvas_grid_number_height, canvas_grid_number_width))

        # 2. compute area according to boundary
        if special_case and wirelen_method == 'manhattan':
            # only mahattan reward have special case
            net_area = abs(matrix_right - matrix_left) * abs(matrix_top - matrix_bottom)
            rc_rudy_per_grid = 0.5
        else:
            net_area = abs(matrix_right - matrix_left) * abs(matrix_top - matrix_bottom)
            rc_rudy_per_grid = net_hpwl / net_area

            if wirelen_method == 'manhattan':
                rc_rudy_per_grid = min(rc_rudy_per_grid, 1)

        # 3. fill rc_rudy into net_congestion_map
        # if not extend
        if len(extended_ratio_to_add) == 1:
            net_congestion_map[matrix_bottom:matrix_top,
                               matrix_left:matrix_right] += rc_rudy_per_grid

        # if extend
        else:
            for extend_n_grids, ratio in enumerate(extended_ratio_to_add):
                net_congestion_map[
                    max(0, matrix_bottom - extend_n_grids):min(canvas_grid_number_height, matrix_top + extend_n_grids),
                    max(0, matrix_left - extend_n_grids):min(canvas_grid_number_width, matrix_right + extend_n_grids)
                ] += rc_rudy_per_grid * ratio

        rudy_values = [rc_rudy_per_grid * (sum(extended_ratio_to_add) - sum(extended_ratio_to_add[:i]))
                       for i in range(len(extended_ratio_to_add))]

        return net_congestion_map, rudy_values, net_area

    def _get_matrix_boundary(self, left, right, bottom, top, canvas_grid_number_width, canvas_grid_number_height):
        matrix_left, matrix_right = math.floor(left), math.ceil(right)
        matrix_bottom, matrix_top = math.floor(bottom), math.ceil(top)
        special_case = False

        # Check I: avoid getting area = 0
        if (matrix_left == matrix_right):
            matrix_left -= 1
            matrix_right += 1
            special_case = True

        if (matrix_bottom == matrix_top):
            matrix_bottom -= 1
            matrix_top += 1
            special_case = True

        # Check II: values should inside board boundary
        matrix_left = max(matrix_left, 0)
        matrix_right = min(matrix_right, canvas_grid_number_width)
        matrix_bottom = max(matrix_bottom, 0)
        matrix_top = min(matrix_top, canvas_grid_number_height)

        # Check III: assertion for bug
        assert matrix_left != matrix_right, "The matrix width of net is 0."
        assert matrix_bottom != matrix_top, "The matrix height of net is 0."

        return matrix_left, matrix_right, matrix_bottom, matrix_top, special_case


def get_naive_reward(
    netlist, placed_nodes_dict, canvas_grid_number_width, canvas_grid_number_height, extended_ratio_to_add,
    wirelen_method, congestion_map_output_file=None, unittest_output_file=None
):
    # compute number of net we have
    total_net_count = 0

    # init `hpwl` and `congestion_map`
    naive_wirelength = 0
    naive_congestion_map = np.zeros((canvas_grid_number_height, canvas_grid_number_width))

    for net in netlist:
        # For unit test results check
        if unittest_output_file is not None:
            print(f"\nNet {net['net_id']}:", file=unittest_output_file)

        net_obj = Net(net, unittest_output_file)

        # compute wirelength and congestion map
        net_count, net_wirelength, net_congestion_map = net_obj.get_net_reward(
            placed_nodes_dict, canvas_grid_number_width, canvas_grid_number_height, extended_ratio_to_add,
            wirelen_method)

        total_net_count += net_count
        naive_wirelength += net_wirelength * net['n_nets'] * net['weight']
        naive_congestion_map += net_congestion_map * net['n_nets']

    # For unit test results check
    if unittest_output_file is not None:
        table = tabulate(naive_congestion_map, headers=list(
            range(naive_congestion_map.shape[1])), showindex="always", floatfmt=".3f")
        print("\nNaive_congestion:\n", table, '\n', file=unittest_output_file)

    # save congestion map to file
    if congestion_map_output_file is not None:
        np.save(congestion_map_output_file, naive_congestion_map)

    # compute RC_rudy
    naive_congestion = cal_rc_rudy(naive_congestion_map, canvas_grid_number_height, canvas_grid_number_width)

    return total_net_count, naive_wirelength, naive_congestion


def cal_rc_rudy(congestion_map, canvas_grid_number_height, canvas_grid_number_width):
    all_RUDY_mean = np.mean(congestion_map)
    top10_RUDY_mean = np.mean(
        sorted(congestion_map.flat, reverse=True)[
            :round(canvas_grid_number_height * canvas_grid_number_width / 10)]
    )
    return top10_RUDY_mean / all_RUDY_mean


def cal_routability(input_data, hwpl, rc, total_net_count, penalty_factor, unittest_output_file=None):
    canvas_width = input_data.metadata['canvas_width']
    canvas_height = input_data.metadata['canvas_height']
    hwpl_std = hwpl / (total_net_count * (canvas_width + canvas_height))
    max_rc = max(1., rc)
    routability = hwpl_std * (1 + penalty_factor * (max_rc - 1.))

    if unittest_output_file is not None:
        print(f"Wirelength: {hwpl} (grid scale: {hwpl / input_data.grid_len})", file=unittest_output_file)
        print(f"  |- HPWL std. term: {total_net_count} x ({canvas_width} + {canvas_height}) = {total_net_count * (canvas_width + canvas_height)}",
              file=unittest_output_file)
        print(f"  |- std. HPWL: {hwpl_std}", file=unittest_output_file)

        print(f"\nCongestion: {rc}", file=unittest_output_file)
        print(f"  |- RC [= max(1, congestion)]: {max_rc}", file=unittest_output_file)

        print(f"\nRoutability: {routability}", file=unittest_output_file)
        print(f"  |- std HPWL * (1 + penalty factor * (rc - 1)) \
                \n     = {np.round(hwpl_std, 5)} * (1 + {penalty_factor} * ({np.round(max_rc, 5)} - 1))\
                \n     = {routability}\n",
              file=unittest_output_file)

    return routability
