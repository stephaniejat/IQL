# DFS to check that it's a valid path.
from typing import List, Optional
<<<<<<< HEAD

import numpy as np

=======
from scipy.linalg import block_diag

import numpy as np

from KRVI.group_rep import GroupRep


>>>>>>> origin/feature/KRVI
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, rotate:int = 0) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    desc = ["".join(x) for x in board]

    for i in range(rotate):
        desc = list(zip(*desc[::-1]))
    return desc

def generate_random_map_with_fixed_lakes(size: int = 8, n_hole: int = 1, rotate: int = 0) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        n_hole: number of lakes to place

    Returns:
        A random valid map, or the empty list [] is none exists
    """
    valid = False
    
    while not valid:
        board = np.full((size, size), "F")  # initialize to make pyright happy
        board[0][0] = "S"
        board[-1][-1] = "G"
        positions = list(range(size**2 - 2)) # number of options for holes
        hole_positions = np.random.choice(positions, size=n_hole, replace = False)
        for position in hole_positions:
            pos_tup = divmod(position + 1, size) # cannot be 0, 0
            board[pos_tup] = "H"
        valid = is_valid(board, size)
    desc = ["".join(x) for x in board]

    for i in range(rotate):
        desc = list(zip(*desc[::-1]))
    return desc
<<<<<<< HEAD
=======


def construct_90deg_block_rot_groups(dim_space: int):
    assert dim_space % 2 == 0
    rotation_matrix = np.array([[0, -1], 
                            [1, 0]])

    blocks = [rotation_matrix]*int(dim_space / 2)
    block_diagonal_matrix = block_diag(*blocks)

    rot_group_2d = [np.eye(2), rotation_matrix, rotation_matrix @ rotation_matrix,  rotation_matrix @ rotation_matrix @ rotation_matrix] 
    rot_group_nd = [np.eye(dim_space), block_diagonal_matrix, block_diagonal_matrix @ block_diagonal_matrix, block_diagonal_matrix @ block_diagonal_matrix @ block_diagonal_matrix]

    return GroupRep(rot_group_nd), GroupRep(rot_group_2d)
>>>>>>> origin/feature/KRVI
