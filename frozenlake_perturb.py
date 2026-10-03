#!/usr/bin/env python3
"""Break the D4 symmetry of a Frozen Lake layout by a controlled amount.

Drop in the root of mtkresearch/IQL, alongside test_rotated_reflected.py.

The wrapper in test_rotated_reflected.py applies one of eight transforms to the
base map at reset: four rotations, and each of those followed by a reflection in
y. That is the dihedral group D4. The invariant kernel asserts the task is
equivariant under it, which holds only while the layout itself is fixed by the
group.

`perturb_map` moves `n_moves` holes onto free tiles, so the layout stops being
D4-symmetric by a controlled, countable amount. n_moves is the violation
severity to sweep: 0 reproduces the paper's setting, larger values make the
assumed invariance progressively false.

`symmetry_orbit_size` counts how many of the eight transforms leave the map
unchanged. 8 means fully symmetric, 1 means every transform gives a distinct
layout, so the invariance is entirely broken. Check this before trusting any
sweep: a perturbation that happens to preserve symmetry is not a violation.

Run directly to self-test:
    python3 frozenlake_perturb.py
"""

from __future__ import annotations

import numpy as np


def as_char_grid(desc) -> np.ndarray:
    grid = np.array(desc)
    if grid.dtype.kind == "S":
        grid = np.char.decode(grid)
    return grid.astype("<U1")


def perturb_map(desc, n_moves: int = 1, rng=None) -> np.ndarray:
    """Move `n_moves` holes to free tiles. Start and goal are never touched."""

    rng = np.random.default_rng(0) if rng is None else rng
    grid = as_char_grid(desc).copy()
    holes = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
             if grid[i, j] == "H"]
    frees = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
             if grid[i, j] == "F"]
    for _ in range(min(n_moves, len(holes), len(frees))):
        h = holes.pop(int(rng.integers(len(holes))))
        f = frees.pop(int(rng.integers(len(frees))))
        grid[h] = "F"
        grid[f] = "H"
    return grid


def d4_variants(grid: np.ndarray) -> list[np.ndarray]:
    """The eight transforms the wrapper applies: rotations and rotate-reflect."""

    rotations = [np.rot90(grid, k) for k in range(4)]
    return rotations + [np.fliplr(r) for r in rotations]


def symmetry_orbit_size(grid: np.ndarray) -> int:
    """How many of the eight transforms leave the layout unchanged."""

    return sum(np.array_equal(grid, v) for v in d4_variants(grid))


def _self_test() -> None:
    base = np.array([list("SFFF"), list("FHFH"), list("FFFH"), list("HFFG")])
    print("base layout, stabiliser size:", symmetry_orbit_size(base))
    for n in (0, 1, 2, 3):
        g = perturb_map(base, n_moves=n, rng=np.random.default_rng(0))
        print(f"n_moves={n}  stabiliser={symmetry_orbit_size(g)}  "
              f"holes={(g == 'H').sum()}")
        print("\n".join("   " + "".join(row) for row in g))


if __name__ == "__main__":
    _self_test()
