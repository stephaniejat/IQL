#!/usr/bin/env python3
"""
Element-by-element audit of the assumed symmetry groups.

Why this exists
---------------
In the wrong-group runs, C4 reached tail regret 0.0025 while the repo's own D4
group reached 0.0738. D4 is a LARGER valid symmetry of a square grid (it adds
the four reflections to C4's four rotations), so it should pool more and do at
least as well. The gap is within one standard error of the D4 run, so it may
simply be noise. But there is a second possibility worth ruling out: that one
of D4's reflection elements does not map actions the way the environment does,
so the kernel pools a pair that is not actually equivalent.

That would matter. The invariant kernel averages the base kernel over every
group element, so a single bad element pollutes every entry of the kernel
matrix, not just the orbits it touches.

What the check does
-------------------
For each group element g and each state-action pair (s, a), it compares

    reward[s, a]   against   reward[g(s), g(a)]

A genuine symmetry gives zero for every pair. Anything non-zero means the
kernel is pooling state-action pairs with different rewards, which is exactly
the false-merge condition the project is about --- except here it would be
accidental rather than deliberate.

It also audits the transition structure, since a symmetry of the reward alone
is not enough: the group must commute with the dynamics for the value function
to be invariant.

Reading the output
------------------
  - Every D4 element OK        -> the symmetry is fine, the C4/D4 gap is noise.
                                  Rerun D4 at more seeds; it should converge
                                  towards C4.
  - Only reflections BAD       -> the environment is rotation-symmetric but not
                                  reflection-symmetric, so the repo's default
                                  group is mildly misspecified here. That is a
                                  real observation, not a bug in your code.
  - A rotation BAD             -> the coordinate or action convention is off,
                                  and both group experiments need rechecking
                                  before any of their numbers are quoted.

Everything is numpy. No torch, no botorch, no training, seconds to run.

Usage
-----
    python3 C4_D4_check.py
    python3 C4_D4_check.py --side 5 --horizon 10
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np


EXPERIMENT_FILE = "approximate_invariance_relevance.py"


def load_experiment_module(path: str):
    """
    Import the experiment file without running its main().

    The experiment file imports the repo module, which pulls in torch and
    gymnasium. We only need its pure-numpy geometry helpers, so we import it
    normally and let it fail loudly if the environment is not set up --- an
    import error here means the audit would be testing different conventions
    from the experiment, which is worse than not running.
    """
    location = pathlib.Path(path)

    if not location.exists():
        sys.exit(
            f"Could not find {path}. Run this from the IQL repo root, "
            f"where the experiment file lives."
        )

    spec = importlib.util.spec_from_file_location("experiment", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment"] = module
    spec.loader.exec_module(module)

    return module


def audit_reward(exp, coords, reward, matrices, label):
    """Check reward[s, a] == reward[g(s), g(a)] for every element and pair."""
    print(f"\n{label}: {len(matrices)} element(s)")

    all_ok = True

    for index, matrix in enumerate(matrices):
        worst = 0.0
        worst_pair = None

        for state in range(len(coords)):
            image_state = exp.nearest_state_index(coords, coords[state] @ matrix.T)

            for action in range(4):
                image_action = exp.transformed_action_index(
                    exp.ACTIONS[action] @ matrix.T
                )

                error = abs(
                    float(reward[state, action] - reward[image_state, image_action])
                )

                if error > worst:
                    worst = error
                    worst_pair = (
                        (state, action),
                        (image_state, image_action),
                    )

        ok = worst < 1e-8
        all_ok = all_ok and ok

        detail = "" if ok else f"  worst: {worst_pair[0]} -> {worst_pair[1]}"

        print(
            f"  [{'OK ' if ok else 'BAD'}] element {index}: "
            f"max reward error {worst:.3e}{detail}"
        )

    return all_ok


def audit_transitions(exp, coords, matrices, label):
    """
    Check that the group commutes with the dynamics.

    A symmetry of the reward is not sufficient. For the value function to be
    invariant we also need g(next(s, a)) == next(g(s), g(a)): moving then
    transforming must equal transforming then moving.
    """
    print(f"\n{label}: transition commutation")

    all_ok = True

    for index, matrix in enumerate(matrices):
        mismatches = 0

        for state in range(len(coords)):
            image_state = exp.nearest_state_index(coords, coords[state] @ matrix.T)

            for action in range(4):
                image_action = exp.transformed_action_index(
                    exp.ACTIONS[action] @ matrix.T
                )

                # transform, then move
                a_then = exp.transition_index(coords, image_state, image_action)

                # move, then transform
                moved = exp.transition_index(coords, state, action)
                b_then = exp.nearest_state_index(coords, coords[moved] @ matrix.T)

                if a_then != b_then:
                    mismatches += 1

        ok = mismatches == 0
        all_ok = all_ok and ok

        print(
            f"  [{'OK ' if ok else 'BAD'}] element {index}: "
            f"{mismatches} non-commuting state-action pair(s)"
        )

    return all_ok


def verdict(d4_reward_ok, d4_transition_ok, matrices):
    print("\n" + "=" * 68)

    if d4_reward_ok and d4_transition_ok:
        print(
            "D4 is a genuine symmetry of this environment.\n"
            "The C4 vs D4 regret gap is therefore not a misspecification:\n"
            "rerun D4 at 12+ seeds and expect it to converge towards C4."
        )

    else:
        # Rotations are the first len(matrices)//2 elements in the repo's
        # construction; reflections follow.
        half = len(matrices) // 2

        print(
            "D4 is NOT a genuine symmetry of this environment as implemented.\n"
            f"Elements 0-{half - 1} are the rotations, {half}-{len(matrices) - 1} "
            "the reflections.\n"
            "If only reflections failed, the repo's default group is mildly\n"
            "misspecified here and C4 is the correct assumption --- report that.\n"
            "If a rotation failed, the coordinate or action convention differs\n"
            "from what the kernel assumes, and the group results need rechecking."
        )

    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--file",
        default=EXPERIMENT_FILE,
        help="Path to the experiment file whose conventions are being audited.",
    )

    parser.add_argument("--side", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)

    args = parser.parse_args()

    exp = load_experiment_module(args.file)

    coords = exp.make_symmetric_grid(args.side)
    reward = exp.baseline_reward(coords)

    print(
        f"Grid {args.side}x{args.side} = {len(coords)} states, "
        f"4 actions, horizon {args.horizon}"
    )

    d4 = exp.repo_d4_matrices()

    d4_reward_ok = audit_reward(
        exp, coords, reward, d4, "repo D4 (rotations + reflections)"
    )
    d4_transition_ok = audit_transitions(exp, coords, d4, "repo D4")

    for n_fold in (4, 3, 6):
        matrices = exp.construct_cyclic_group(n_fold)
        label = f"C{n_fold}"

        audit_reward(exp, coords, reward, matrices, label)
        audit_transitions(exp, coords, matrices, label)

    verdict(d4_reward_ok, d4_transition_ok, d4)


if __name__ == "__main__":
    main()


# Output:
# device cpu
# Grid 5x5 = 25 states, 4 actions, horizon 10

# repo D4 (rotations + reflections): 8 element(s)
#   [OK ] element 0: max reward error 0.000e+00
#   [OK ] element 1: max reward error 0.000e+00
#   [OK ] element 2: max reward error 0.000e+00
#   [OK ] element 3: max reward error 0.000e+00
#   [OK ] element 4: max reward error 0.000e+00
#   [OK ] element 5: max reward error 0.000e+00
#   [OK ] element 6: max reward error 0.000e+00
#   [OK ] element 7: max reward error 0.000e+00

# repo D4: transition commutation
#   [OK ] element 0: 0 non-commuting state-action pair(s)
#   [OK ] element 1: 0 non-commuting state-action pair(s)
#   [OK ] element 2: 0 non-commuting state-action pair(s)
#   [OK ] element 3: 0 non-commuting state-action pair(s)
#   [OK ] element 4: 0 non-commuting state-action pair(s)
#   [OK ] element 5: 0 non-commuting state-action pair(s)
#   [OK ] element 6: 0 non-commuting state-action pair(s)
#   [OK ] element 7: 0 non-commuting state-action pair(s)

# C4: 4 element(s)
#   [OK ] element 0: max reward error 0.000e+00
#   [OK ] element 1: max reward error 0.000e+00
#   [OK ] element 2: max reward error 0.000e+00
#   [OK ] element 3: max reward error 0.000e+00

# C4: transition commutation
#   [OK ] element 0: 0 non-commuting state-action pair(s)
#   [OK ] element 1: 0 non-commuting state-action pair(s)
#   [OK ] element 2: 0 non-commuting state-action pair(s)
#   [OK ] element 3: 0 non-commuting state-action pair(s)

# C3: 3 element(s)
#   [OK ] element 0: max reward error 0.000e+00
#   [BAD] element 1: max reward error 7.500e-01  worst: (0, 0) -> (9, 3)
#   [BAD] element 2: max reward error 7.500e-01  worst: (0, 0) -> (21, 1)

# C3: transition commutation
#   [OK ] element 0: 0 non-commuting state-action pair(s)
#   [BAD] element 1: 52 non-commuting state-action pair(s)
#   [BAD] element 2: 44 non-commuting state-action pair(s)

# C6: 6 element(s)
#   [OK ] element 0: max reward error 0.000e+00
#   [BAD] element 1: max reward error 7.500e-01  worst: (0, 0) -> (3, 3)
#   [BAD] element 2: max reward error 7.500e-01  worst: (0, 0) -> (9, 3)
#   [OK ] element 3: max reward error 0.000e+00
#   [BAD] element 4: max reward error 7.500e-01  worst: (0, 0) -> (21, 1)
#   [BAD] element 5: max reward error 7.500e-01  worst: (0, 0) -> (15, 1)

# C6: transition commutation
#   [OK ] element 0: 0 non-commuting state-action pair(s)
#   [BAD] element 1: 44 non-commuting state-action pair(s)
#   [BAD] element 2: 52 non-commuting state-action pair(s)
#   [OK ] element 3: 0 non-commuting state-action pair(s)
#   [BAD] element 4: 44 non-commuting state-action pair(s)
#   [BAD] element 5: 44 non-commuting state-action pair(s)

# ====================================================================
# D4 is a genuine symmetry of this environment.
# The C4 vs D4 regret gap is therefore not a misspecification:
# rerun D4 at 12+ seeds and expect it to converge towards C4.
# ====================================================================
