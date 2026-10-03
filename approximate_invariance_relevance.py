#!/usr/bin/env python3
"""
Controlled approximate-invariance experiment using the repo-native KRVI implementation.

This file:
  1. imports KRVI, InvariantKernel and RBFKernel from
     KRVI_algo_test_rotated_invariant_fulloptim.py;
  2. gives that exact KRVI implementation a small D4-symmetric grid environment;
  3. perturbs one state-action reward by delta;
  4. captures the quantities needed for the proposal:
       - tail episodic regret,
       - whether the optimal decision changes,
       - the model's predicted Q-gap across an orbit,
       - log-marginal-likelihood ratio on the same Bellman data;
  5. leaves the actual KRVI GP/value-iteration/training machinery to the repo.

Modes:

    exact
        No structural violation.

    relevant
        Perturb one state-action selected by the previous relevance heuristic.

    irrelevant
        Perturb a strongly suboptimal state-action, intended to remain
        decision-irrelevant over the supplied delta range.

    flip
        Choose the state with the smallest positive action margin and perturb
        its runner-up action. The delta sweep is defined relative to that
        margin so that the sweep explicitly brackets the point where the
        greedy optimal action changes.

For the flip experiment:

    python3 approximate_invariance_relevance.py \
        --mode flip \
        --H 10 --T 200 --seeds 5 --beta 0.1 \
        --out flip.json

The flip sweep is automatically constructed from the selected baseline
action margin:

    0,
    0.25 m,
    0.5 m,
    0.9 m,
    1.1 m,
    1.5 m,
    2.0 m

where m is the baseline Q-margin between the optimal action and runner-up.

The intended prediction is that the optimal action changes between 0.9m and
1.1m. The JSON records the actual decision change, so this is checked rather
than assumed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import types

import numpy as np


# ---------------------------------------------------------------------------
# The repo currently calls wandb.Settings(start_method="thread"), which is
# incompatible with newer W&B versions. We do not need logging for this
# experiment, so W&B is stubbed BEFORE importing the repo module.
# ---------------------------------------------------------------------------

# We import the real W&B only when the user asks for logging, because the repo
# calls wandb.Settings(start_method="thread"), which newer versions reject. The
# repo's own KRVI is always constructed with logging=None here, so it never
# touches W&B; this file does its own logging instead.

WANDB_REQUESTED = any(a.startswith("--wandb_project") for a in sys.argv)

if WANDB_REQUESTED:
    import wandb  # noqa: F401  (real module)
elif "wandb" not in sys.modules:
    stub = types.ModuleType("wandb")
    stub.init = lambda *a, **k: None
    stub.log = lambda *a, **k: None
    stub.Image = lambda *a, **k: None
    stub.run = types.SimpleNamespace(summary={})
    sys.modules["wandb"] = stub


# ---------------------------------------------------------------------------
# Import the actual repo implementation.
# ---------------------------------------------------------------------------

import KRVI_algo_test_rotated_invariant_fulloptim as repo

# Selectable acquisition rule (ucb / greedy / ts) and the orbit-residual
# diagnostic. Imported after the W&B stub above so it picks up the same module.
from ts_acquisition import (
    ACQUISITIONS,
    AcquisitionKRVI,
    orbit_residual_contrast,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise RuntimeError(
        "This script needs gymnasium because the repo's KRVI expects a "
        "gym-style environment."
    ) from exc

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood


# ---------------------------------------------------------------------------
# Symmetric environment
# ---------------------------------------------------------------------------

ACTIONS = np.array(
    [
        [-1.0, 0.0],  # left
        [0.0, -1.0],  # down
        [1.0, 0.0],  # right
        [0.0, 1.0],  # up
    ],
    dtype=np.float64,
)


def action_to_vec(action: int) -> np.ndarray:
    """Same 2-D action representation used by the invariant kernel."""
    return ACTIONS[int(action)].copy()


class SymmetricGridEnv(gym.Env):
    """
    Small deterministic grid whose dynamics are exactly D4-equivariant.

    States are 2-D coordinates in [-1, 1]^2.
    Actions are the four compass directions.
    Boundary moves stay in place.

    The reward is supplied as a state-action matrix. The baseline reward is
    D4-symmetric; controlled violations are inserted into that matrix.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward: np.ndarray,
        coords: np.ndarray,
        rng: np.random.Generator,
    ):
        super().__init__()

        self.reward = np.asarray(reward, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(self.coords)
        self.rng = rng

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

        self._state = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Uniform start distribution is itself D4-symmetric.
        self._state = int(self.rng.integers(self.n))

        return self.coords[self._state].copy(), {}

    def step(self, action):
        action = int(action)
        xy = self.coords[self._state]

        # Find the nearest grid coordinate after the move.
        proposed = xy + ACTIONS[action] * (2.0 / (len_side(self.coords) - 1))
        proposed = np.clip(proposed, -1.0, 1.0)

        next_state = nearest_state_index(self.coords, proposed)
        reward = float(self.reward[self._state, action])

        self._state = next_state

        # No terminal states: the finite horizon belongs to KRVI.
        return (
            self.coords[next_state].copy(),
            reward,
            False,
            False,
            {},
        )


def len_side(coords: np.ndarray) -> int:
    """Number of points on one coordinate axis."""
    return int(round(np.sqrt(len(coords))))


def nearest_state_index(
    coords: np.ndarray,
    point: np.ndarray,
) -> int:
    d2 = np.sum(
        (coords - point.reshape(1, 2)) ** 2,
        axis=1,
    )
    return int(np.argmin(d2))


def make_symmetric_grid(side: int = 5) -> np.ndarray:
    """
    Square coordinate grid centred at zero.

    The ordering is row-major but all group operations are performed through
    coordinates, not through assumptions about indices.
    """
    vals = np.linspace(-1.0, 1.0, side)

    return np.array(
        [[x, y] for x in vals for y in vals],
        dtype=np.float64,
    )


def transition_index(
    coords: np.ndarray,
    state_idx: int,
    action_idx: int,
) -> int:
    side = len_side(coords)
    step = 2.0 / (side - 1)

    proposed = coords[state_idx] + ACTIONS[action_idx] * step
    proposed = np.clip(proposed, -1.0, 1.0)

    return nearest_state_index(coords, proposed)


def baseline_reward(coords: np.ndarray) -> np.ndarray:
    """
    D4-invariant state-action reward.

    Reward is highest near the origin. It is intentionally simple: the
    experiment is about the learner's structural assumption, not about making
    a complicated benchmark.
    """
    dist2 = np.sum(coords**2, axis=1)
    state_reward = 1.0 - dist2

    return np.repeat(
        state_reward[:, None],
        4,
        axis=1,
    )


# ---------------------------------------------------------------------------
# Exact finite-horizon dynamic programming for the benchmark.
# ---------------------------------------------------------------------------


def optimal_values(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
):
    """
    Return V_0 and Q_0 for the finite-horizon deterministic MDP.

    V_0[s] is the optimal total reward over `horizon` steps starting at s.
    """
    n = len(coords)

    V_next = np.zeros(n, dtype=np.float64)
    Q0 = np.zeros((n, 4), dtype=np.float64)

    # Work backwards.
    for h in reversed(range(horizon)):
        Q = np.empty((n, 4), dtype=np.float64)

        for s in range(n):
            for a in range(4):
                ns = transition_index(coords, s, a)
                Q[s, a] = reward[s, a] + V_next[ns]

        V_next = np.max(Q, axis=1)
        Q0 = Q

    return V_next, Q0


def optimal_q_by_step(coords, reward, horizon):
    """
    Optimal values at EVERY step of the episode, not only the first.

    V[h, s] and Q[h, s, a] are the best achievable total reward from step h
    onward (so H - h steps remain). optimal_values returns only h = 0; the
    regret decomposition below needs every step.
    """
    n = len(coords)
    V = np.zeros((horizon + 1, n), dtype=np.float64)
    Q = np.zeros((horizon, n, 4), dtype=np.float64)

    for h in reversed(range(horizon)):
        for s in range(n):
            for a in range(4):
                Q[h, s, a] = reward[s, a] + V[h + 1, transition_index(coords, s, a)]

        V[h] = Q[h].max(axis=1)

    return V, Q


def decompose_regret(coords, reward, horizon, episodes, cell, tail_n):
    """
    Split each episode's regret by WHERE it was paid.

    Because the moves are deterministic, an episode's regret is exactly the
    sum over its steps of how much worse the chosen action was than the best
    one at that step:

        regret = V_0(s_0) - sum_h r(s_h, a_h) = sum_h [ V_h(s_h) - Q_h(s_h, a_h) ]

    (each Q_h(s_h, a_h) = r(s_h, a_h) + V_{h+1}(s_{h+1}), so the sum
    telescopes). `telescoping_error` reports the largest mismatch between the
    two sides, which should be ~0; if it is not, the reconstruction is wrong
    and nothing else here should be trusted.

    Every unit of per-step loss is attributed to the state it was paid at:

        violated  the state carrying the violation
        orbit     the other states in the violated cell's symmetry orbit. The
                  environment did not change there, so a plain kernel has
                  nothing new to learn. An invariant kernel, which cannot tell
                  these cells apart from the violated one, should copy the
                  extra reward onto them and start choosing the mirrored
                  action. Loss paid here is a cost of POOLING, not of the
                  violation itself.
        rest      everything else

    leak_rate is how often, when at an orbit state, the learner took the
    mirror-image of the violated action. fix_rate is how often, at the
    violated state, it took the violated action (the new best action once
    delta exceeds the margin).

    Computed on the last `tail_n` recorded episodes, matching tail regret.
    """
    V, Q = optimal_q_by_step(coords, reward, horizon)

    vs, va = cell
    orbit_pairs = {
        (int(s_), int(a_))
        for (s_, a_) in full_orbit(coords, vs, va)
        if int(s_) != int(vs)
    }
    orbit_states = {s_ for (s_, _) in orbit_pairs}

    tail = episodes[-tail_n:] if tail_n else episodes

    loss_violated, loss_orbit, loss_rest = [], [], []
    orbit_visits = orbit_leaks = 0
    violated_visits = violated_fixes = 0
    worst_mismatch = 0.0

    for ep in tail:
        state = int(ep["start"])
        lv = lo = lr = 0.0

        for h, action in enumerate(ep["actions"]):
            if h >= horizon:
                break

            action = int(action)
            gap = float(V[h, state] - Q[h, state, action])

            if state == vs:
                lv += gap
                violated_visits += 1
                violated_fixes += int(action == va)
            elif state in orbit_states:
                lo += gap
                orbit_visits += 1
                orbit_leaks += int((state, action) in orbit_pairs)
            else:
                lr += gap

            state = transition_index(coords, state, action)

        regret = float(V[0, int(ep["start"])] - sum(ep["rewards"][:horizon]))
        worst_mismatch = max(worst_mismatch, abs(regret - (lv + lo + lr)))

        loss_violated.append(lv)
        loss_orbit.append(lo)
        loss_rest.append(lr)

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "loss_violated": mean(loss_violated),
        "loss_orbit": mean(loss_orbit),
        "loss_rest": mean(loss_rest),
        "leak_rate": orbit_leaks / orbit_visits if orbit_visits else float("nan"),
        "fix_rate": violated_fixes / violated_visits
        if violated_visits
        else float("nan"),
        "orbit_visits_per_episode": orbit_visits / max(1, len(tail)),
        "violated_visits_per_episode": violated_visits / max(1, len(tail)),
        "telescoping_error": worst_mismatch,
        "orbit_size": len(orbit_pairs),
        # Raw counts, never None. leak_rate and fix_rate above are ratios and
        # go None when their denominator is zero, which is precisely the case
        # a non-optimistic acquisition rule is expected to produce -- so the
        # coverage question has to be answerable from counts, not rates.
        # violated_pair_visits counts the violated (state, ACTION) pair, which
        # is the runner-up action an optimistic learner never takes; this is
        # the quantity "did posterior sampling ever collect data there".
        "violated_state_visits": violated_visits,
        "violated_pair_visits": violated_fixes,
        "orbit_state_visits": orbit_visits,
        "orbit_pair_visits": orbit_leaks,
        "episodes_counted": len(tail),
    }


# ---------------------------------------------------------------------------
# D4 group and orbit helpers
# ---------------------------------------------------------------------------


def repo_d4_matrices():
    """
    Return the exact 8 transformations used by the repo kernel.

    This mirrors invariant_kernel_fixed.py::construct_rot_and_reflection_group:

      - 4 rotations: 0, 90, 180, 270 degrees
      - 4 rotations followed by reflection across the x-axis

    The repo applies these to row vectors as x @ g.T. We use the same
    convention here.
    """
    rot_90 = np.array(
        [
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    reflect_x = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    rotations = [np.eye(2)]
    current = np.eye(2)

    for _ in range(3):
        current = current @ rot_90
        rotations.append(current.copy())

    return rotations + [R @ reflect_x for R in rotations]


def construct_cyclic_group(n_fold: int):
    """
    Cyclic rotation group C_n on (x, y), in the repo's row-vector convention.

    The repo's kernel uses D4: four 90-degree rotations plus their reflections,
    a genuine symmetry of a square grid. n_fold in {1, 2, 4} therefore pools
    orbits that really are equivalent; any other n_fold rotates by an angle
    that is not a symmetry of the grid, so the kernel pools points with
    different values while the environment stays exactly symmetric.

    That separates two situations the proposal needs to keep apart:
      - the world is only approximately symmetric (a reward violation), and
      - the learner assumed the wrong symmetry (this function).

    |C_n| = n, so group order varies with n_fold. Report it: a larger invalid
    group pools more and should do more damage.
    """
    matrices = []

    for k in range(n_fold):
        theta = 2.0 * np.pi * k / n_fold
        c, s_ = np.cos(theta), np.sin(theta)
        matrices.append(np.array([[c, s_], [-s_, c]]))

    return matrices


def make_cyclic_transform(n_fold: int):
    """
    Drop-in replacement for repo.apply_rotation_group with a chosen group.

    Mirrors invariant_kernel_fixed.apply_rotation_group: block diagonal over
    (x, y) pairs, applied as x @ g.T, stacked on dim -3.
    """
    import torch
    from scipy.linalg import block_diag

    base = construct_cyclic_group(n_fold)

    def apply(x):
        pairs = x.shape[-1] // 2

        group = [
            torch.tensor(
                block_diag(*([M] * pairs)),
                dtype=torch.float32,
                device=x.device,
            )
            for M in base
        ]

        return torch.stack([x @ g.T for g in group], dim=-3)

    return apply


def transformed_action_index(
    action_vec: np.ndarray,
) -> int:
    """Map a transformed compass action back to one of four action ids."""
    d2 = np.sum(
        (ACTIONS - action_vec.reshape(1, 2)) ** 2,
        axis=1,
    )

    return int(np.argmin(d2))


def group_is_valid_for(coords, reward, n_fold, tol: float = 1e-8):
    """
    Check numerically whether C_n really is a symmetry of this reward.

    Without this the wrong-group condition is an assumption rather than a
    manipulation: the claim that C_3 pools non-equivalent points has to be
    verified, not asserted. Returns (is_valid, max_error).
    """
    if n_fold is None:
        return True, 0.0

    worst = 0.0

    for M in construct_cyclic_group(n_fold):
        for s in range(len(coords)):
            ts = nearest_state_index(coords, coords[s] @ M.T)

            for a in range(4):
                ta = transformed_action_index(ACTIONS[a] @ M.T)
                worst = max(worst, abs(float(reward[s, a] - reward[ts, ta])))

    return bool(worst <= tol), float(worst)


def full_orbit(
    coords: np.ndarray,
    state_idx: int,
    action_idx: int,
):
    """Return the full orbit under the repo's 8-element D4 group."""
    x = coords[state_idx]
    a = ACTIONS[action_idx]

    orbit = []

    for G in repo_d4_matrices():
        # Match invariant_kernel_fixed.py exactly: x @ G.T.
        sx = x @ G.T
        av = a @ G.T

        si = nearest_state_index(coords, sx)
        ai = transformed_action_index(av)

        orbit.append((si, ai))

    return list(dict.fromkeys(orbit))


def rotate180_state_partner(
    coords: np.ndarray,
    state_idx: int,
) -> int:
    """State partner under the 180-degree D4 transformation."""
    target = -coords[state_idx]

    return nearest_state_index(coords, target)


def opposite_action(action_idx: int) -> int:
    """Action partner under the 180-degree D4 transformation."""
    return {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }[int(action_idx)]


# ---------------------------------------------------------------------------
# Controlled violations
# ---------------------------------------------------------------------------


def choose_violation_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
):
    """
    Put the point violation at a genuinely decision-relevant state-action:
    the baseline optimal Q_0 maximum, rather than merely the largest reward.
    """
    _, Q0 = optimal_values(
        coords,
        reward,
        horizon,
    )

    s, a = np.unravel_index(
        int(np.argmax(Q0)),
        Q0.shape,
    )

    return int(s), int(a)


def choose_relevant_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
):
    """
    Choose a state-action pair where a perturbation is most likely to be
    decision-relevant.

    This is retained for comparison with the previous experiment.

    Note that perturbing the optimal action upward cannot itself cause the
    greedy action at that state to flip.
    """
    _, Q0 = optimal_values(
        coords,
        reward,
        horizon,
    )

    margins = []

    for s in range(len(coords)):
        order = np.sort(Q0[s])
        margin = order[-1] - order[-2]
        margins.append(margin)

    s = int(np.argmax(margins))
    a = int(np.argmax(Q0[s]))

    return s, a


def choose_irrelevant_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
    relevant_cell: tuple[int, int],
):
    """
    Choose a state-action pair whose perturbation is unlikely to affect the
    optimal decision.

    We deliberately choose a non-optimal action with a large baseline
    suboptimality gap.

    If the candidate is in the same orbit as the relevant cell, keep searching
    until we find a distinct orbit.
    """
    _, Q0 = optimal_values(
        coords,
        reward,
        horizon,
    )

    relevant_orbit = set(
        full_orbit(
            coords,
            *relevant_cell,
        )
    )

    candidates = []

    for s in range(len(coords)):
        best = float(np.max(Q0[s]))

        for a in range(4):
            if (s, a) in relevant_orbit:
                continue

            gap = best - Q0[s, a]

            if gap > 1e-8:
                candidates.append((gap, s, a))

    if not candidates:
        raise RuntimeError("Could not find a distinct suboptimal state-action pair.")

    # Largest suboptimality gap gives the most room for a violation without
    # changing the optimal action.
    _, s, a = max(candidates)

    return int(s), int(a)


def optimal_occupancy(coords, reward, horizon):
    """
    Visit frequency of each state under the optimal policy, from a uniform
    start distribution (which is what SymmetricGridEnv.reset samples).

    Episodic regret averages return over episodes, so a violation only costs
    anything if the optimal policy actually passes through the perturbed
    state often enough for the changed action to be taken. A sup over a
    near-optimal set charges for states the policy never reaches; this
    weights by where it goes.

    Returns an array summing to 1 over states.
    """
    _, Q0 = optimal_values(coords, reward, horizon)
    policy = np.argmax(Q0, axis=1)

    n = len(coords)
    counts = np.zeros(n, dtype=np.float64)

    for start in range(n):
        s_ = start
        for _ in range(horizon):
            counts[s_] += 1.0
            s_ = transition_index(coords, s_, int(policy[s_]))

    total = counts.sum()

    return counts / total if total > 0 else counts


def choose_flip_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
    min_margin: float = 1e-6,
    occupancy_rank: str = "any",
):
    """
    Choose a state-action pair at which a swept violation will cross the point
    where the optimal action changes.

    The important correction relative to choose_relevant_cell is:

        DO NOT perturb the optimal action.

    Instead:

        1. find the state with the smallest positive action margin;
        2. identify its runner-up action;
        3. perturb that runner-up action upward.

    At the local one-step level, the runner-up Q value increases by delta, so
    the greedy decision is expected to cross when delta reaches the baseline
    action margin.

    The actual finite-horizon dynamic program is recomputed after perturbation,
    so `decision_changed` in the experiment is the authoritative check.

    Self-paired cells are skipped: if the orbit has only one member there is
    no nontrivial equivalence to break.

    occupancy_rank controls WHERE the flip is placed:

        "any"   smallest positive margin anywhere (original behaviour)
        "high"  smallest margin among the most-visited states under the
                optimal policy
        "low"   smallest margin among the least-visited states

    Episodic regret averages return over episodes, so a changed action only
    costs anything if the optimal policy is actually at that state often
    enough. "high" against "low" is therefore the manipulation that separates
    local decision relevance from return relevance.

    Returns:

        (cell, margin, baseline_optimal_action)

    where:

        cell = (state_index, runner_up_action)
        margin = Q(best) - Q(runner_up)
        baseline_optimal_action = best action before perturbation
    """
    _, Q0 = optimal_values(
        coords,
        reward,
        horizon,
    )

    occupancy = optimal_occupancy(coords, reward, horizon)

    if occupancy_rank == "any":
        allowed_states = set(range(len(coords)))
    else:
        by_visits = np.argsort(occupancy)

        # Top or bottom third, so the contrast is not one state wide.
        k = max(1, len(coords) // 3)

        allowed_states = {
            int(x)
            for x in (by_visits[-k:] if occupancy_rank == "high" else by_visits[:k])
        }

    candidates = []

    for s in range(len(coords)):
        order = np.argsort(Q0[s])

        best_a = int(order[-1])
        runner_a = int(order[-2])

        margin = float(Q0[s, best_a] - Q0[s, runner_a])

        if margin <= min_margin:
            continue

        if s not in allowed_states:
            continue

        # The selected state-action must have a nontrivial orbit.
        orbit = full_orbit(
            coords,
            s,
            runner_a,
        )

        if len(set(orbit)) < 2:
            continue

        candidates.append(
            (
                margin,
                s,
                runner_a,
                best_a,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"No breakable positive-margin state found with "
            f"occupancy_rank={occupancy_rank!r}."
        )

    # Smallest positive margin gives the earliest controlled crossing.
    margin, s, runner_a, best_a = min(
        candidates,
        key=lambda x: x[0],
    )

    return (
        (int(s), int(runner_a)),
        float(margin),
        int(best_a),
        float(occupancy[s]),
    )


def violate(
    coords: np.ndarray,
    reward: np.ndarray,
    delta: float,
    mode: str,
    horizon: int,
    flip_cell: tuple[int, int] | None = None,
):
    """
    Construct one of four conditions:

      exact
          no structural violation;

      relevant
          break an equivalence at a decision-relevant location using the
          original relevance heuristic;

      irrelevant
          break an equivalence at a location where the perturbed action is
          intended to remain suboptimal;

      flip
          perturb the runner-up action at a state with a small positive
          action margin so that the delta sweep crosses the decision boundary.

    The perturbation is applied to exactly one member of a D4 orbit.
    """
    out = reward.copy()

    if mode == "exact":
        cell = choose_relevant_cell(
            coords,
            reward,
            horizon,
        )
        return out, cell

    relevant_cell = choose_relevant_cell(
        coords,
        reward,
        horizon,
    )

    if mode == "relevant":
        cell = relevant_cell

    elif mode == "flip":
        if flip_cell is None:
            cell, _, _ = choose_flip_cell(
                coords,
                reward,
                horizon,
            )
        else:
            cell = flip_cell

    elif mode == "irrelevant":
        cell = choose_irrelevant_cell(
            coords,
            reward,
            horizon,
            relevant_cell,
        )

    else:
        raise ValueError(f"unknown violation mode: {mode}")

    out[cell] += delta

    return out, cell


# ---------------------------------------------------------------------------
# Repo KRVI probe
# ---------------------------------------------------------------------------


class ProbedKRVI(AcquisitionKRVI):
    """
    Thin subclass: the learning algorithm remains the repo's KRVI.

    AcquisitionKRVI is repo.KRVI with the acquisition rule made selectable;
    with acquisition="ucb" (the default) it is behaviourally the repo method.

    The only addition is recording the GP/data produced by
    GP_regression_torch so we can calculate diagnostics after train().
    """

    def __init__(self, *args, **kwargs):
        self.gp_calls = []
        self.last_gp = None
        self.last_X = None
        self.last_y = None

        super().__init__(
            *args,
            **kwargs,
        )

    def GP_regression_torch(self, X, y):
        gp = super().GP_regression_torch(
            X,
            y,
        )

        # Copy the data immediately: the repo's training loop reuses arrays.
        self.gp_calls.append(
            {
                "gp": gp,
                "X": np.asarray(X).copy(),
                "y": np.asarray(y).copy(),
            }
        )

        self.last_gp = gp
        self.last_X = np.asarray(X).copy()
        self.last_y = np.asarray(y).copy()

        return gp


def make_repo_kernel(kind: str, n_fold=None):
    """n_fold=None uses the repo's own D4 group; an int uses C_n instead."""
    if kind == "invariant":
        transform = (
            repo.apply_rotation_group
            if n_fold is None
            else make_cyclic_transform(n_fold)
        )

        return repo.InvariantKernel(
            base_kernel=repo.RBFKernel(),
            transformations=transform,
            is_isotropic=True,
            is_group=True,
        )

    if kind == "rbf":
        return repo.RBFKernel()

    raise ValueError(f"unknown kernel kind: {kind}")


def make_repo_krvi(
    kernel,
    env,
    beta: float,
    horizon: int,
    length_scale: float,
    noise: float,
    seed: int,
    acquisition: str = "ucb",
    coords=None,
):
    """
    Instantiate KRVI while passing only constructor arguments that the local
    copy of the repo actually accepts.

    acquisition / coords bypass that filter: they belong to AcquisitionKRVI,
    not to repo.KRVI, so they are not in the repo signature. The default
    "ucb" keeps every existing call site (notably fit_other_kernel_lml, which
    only fits a GP and never trains) behaviourally unchanged.
    """
    candidate = {
        "kernel": kernel,
        "env": env,
        "beta": beta,
        "horizon": horizon,
        "action_transformation": action_to_vec,
        "len_scale": length_scale,
        "noise_reg": noise,
        "optim_botorch": 0,
        "iterations": 0,
        "seed": seed,
        "logging": None,
    }

    sig = inspect.signature(repo.KRVI.__init__)

    accepted = {
        name: value for name, value in candidate.items() if name in sig.parameters
    }

    extra = {"acquisition": acquisition}

    if acquisition == "ts":
        extra["coords"] = coords

    return ProbedKRVI(**accepted, **extra)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def gp_lml(gp) -> float:
    """
    Exact GPyTorch log marginal likelihood of the repo GP model.

    No sklearn GP is introduced here.
    """
    gp.eval()
    gp.likelihood.eval()

    with torch.no_grad():
        output = gp(*gp.train_inputs)

        mll = ExactMarginalLogLikelihood(
            gp.likelihood,
            gp,
        )

        return float(
            mll(
                output,
                gp.train_targets,
            ).item()
        )


def fit_other_kernel_lml(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    env,
    beta: float,
    horizon: int,
    length_scale: float,
    noise: float,
    seed: int,
    n_fold=None,
) -> float:
    """
    Fit the repo's GP implementation with the other kernel on exactly the
    same Bellman regression data.

    This is the passive-data model comparison: the data are held fixed while
    only the structural kernel changes.
    """
    probe = make_repo_krvi(
        make_repo_kernel(kind, n_fold),
        env,
        beta,
        horizon,
        length_scale,
        noise,
        seed,
    )

    gp = probe.GP_regression_torch(
        X,
        y,
    )

    return gp_lml(gp)


def predict_orbit_gap(
    probe: ProbedKRVI,
    coords: np.ndarray,
    cell: tuple[int, int],
) -> float | None:
    if probe.last_gp is None:
        return None

    s_idx, a_idx = cell

    state = coords[s_idx]
    action = action_to_vec(a_idx)

    partner_state = -state
    partner_action = -action

    states = np.vstack(
        [
            state,
            partner_state,
        ]
    )

    actions = np.vstack(
        [
            action,
            partner_action,
        ]
    )

    mean, _, _ = probe.predict_with_gp(
        probe.last_gp,
        states,
        actions,
    )

    mean = np.asarray(mean).reshape(-1)

    return float(abs(mean[0] - mean[1]))


def run_one(
    coords,
    reward,
    horizon,
    T,
    beta,
    noise,
    kind,
    length_scale,
    seed,
    cell,
    n_fold=None,
    acquisition="ucb",
):
    env = SymmetricGridEnv(
        reward=reward,
        coords=coords,
        rng=np.random.default_rng(seed),
    )

    probe = make_repo_krvi(
        make_repo_kernel(kind, n_fold),
        env,
        beta,
        horizon,
        length_scale,
        noise,
        seed,
        acquisition=acquisition,
        coords=coords,
    )

    # Actual repo KRVI training loop.
    probe.train(T)

    episodes = getattr(
        env,
        "episodes",
        [],
    )

    # Final GP/value model.
    orbit_gap = predict_orbit_gap(
        probe,
        coords,
        cell,
    )

    # Orbit residual contrast on the per-step models of the final episode.
    # gp_calls accumulates one entry per (episode, h); the last `horizon`
    # entries are the h = H-1 ... 0 models fitted on everything collected, so
    # together they cover every step of the state-action space the agent
    # actually visited. last_gp alone would only cover h = 0.
    orbit_residual = orbit_residual_contrast(
        probe.gp_calls[-horizon:],
        coords,
        cell,
        full_orbit(coords, int(cell[0]), int(cell[1])),
        action_to_vec,
    )

    # LML on the actual model/data used by this KRVI run.
    lml_same_kernel = gp_lml(probe.last_gp) if probe.last_gp is not None else None

    # Same data, other kernel.
    other = "rbf" if kind == "invariant" else "invariant"

    other_lml = (
        fit_other_kernel_lml(
            probe.last_X,
            probe.last_y,
            other,
            env,
            beta,
            horizon,
            length_scale,
            noise,
            seed,
            n_fold,
        )
        if probe.last_X is not None
        else None
    )

    # Exact finite-horizon optimal value for the same violated reward.
    V_star, _ = optimal_values(
        coords,
        reward,
        horizon,
    )

    regrets = []
    trajectories = []

    for ep in episodes:
        start = ep["start"]
        total_reward = sum(ep["rewards"])

        regrets.append(float(V_star[start] - total_reward))

        trajectories.append(tuple(ep["actions"]))

    return {
        "regret": np.asarray(
            regrets,
            dtype=float,
        ),
        "trajectory": tuple(trajectories),
        "orbit_gap": orbit_gap,
        "lml": {
            "invariant": (lml_same_kernel if kind == "invariant" else other_lml),
            "rbf": (lml_same_kernel if kind == "rbf" else other_lml),
        },
        "last_X": probe.last_X,
        "last_y": probe.last_y,
        "episodes": list(episodes),
        "orbit_residual": orbit_residual,
        "ts_fallbacks": int(getattr(probe, "ts_fallbacks", 0)),
    }


# ---------------------------------------------------------------------------
# Add recording to the environment without touching KRVI.
# ---------------------------------------------------------------------------

_original_reset = SymmetricGridEnv.reset
_original_step = SymmetricGridEnv.step


def _recording_step(
    self,
    action,
):
    result = _original_step(
        self,
        action,
    )

    _, reward, _, _, _ = result

    self._episode_actions.append(int(action))

    self._episode_rewards.append(float(reward))

    return result


def _recording_reset_and_archive(
    self,
    *args,
    **kwargs,
):
    # Archive the previous episode before beginning a new one.
    if hasattr(self, "_episode_actions") and self._episode_actions:
        if not hasattr(self, "episodes"):
            self.episodes = []

        self.episodes.append(
            {
                "start": self._episode_start,
                "actions": list(self._episode_actions),
                "rewards": list(self._episode_rewards),
            }
        )

    result = _original_reset(
        self,
        *args,
        **kwargs,
    )

    self._episode_start = int(self._state)

    self._episode_actions = []
    self._episode_rewards = []

    if not hasattr(self, "episodes"):
        self.episodes = []

    return result


# KRVI calls reset at the beginning of every episode, so archive on reset.
SymmetricGridEnv.reset = _recording_reset_and_archive
SymmetricGridEnv.step = _recording_step


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def stderr(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        float(
            np.std(
                x,
                ddof=1,
            )
            / np.sqrt(len(x))
        )
        if len(x) > 1
        else float("nan")
    )


class Logger:
    """
    Live W&B logging, in the same style as the repo's KRVI.train loop:
    config at init, metrics per step, summary at the end, artefacts attached.

    A no-op when --wandb_project is not given, so the JSON and PNG outputs are
    produced identically either way.
    """

    def __init__(self, args, config):
        self.enabled = bool(args.wandb_project)
        self.run = None

        if not self.enabled:
            return

        import wandb

        name = args.wandb_name or (
            f"{'curve' if args.curve else args.mode}"
            f"-{'D4' if args.n_fold is None else f'C{args.n_fold}'}"
            f"-{args.acquisition}"
            + (
                ""
                if args.occupancy_rank == "any"
                else f"-{args.occupancy_rank}occ"
            )
        )

        self.run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=name,
            reinit=True,
            config=config,
        )

    def log(self, metrics, step=None):
        if self.run is not None:
            self.run.log(metrics, step=step)

    def summary(self, key, value):
        if self.run is not None:
            self.run.summary[key] = value

    def save(self, path):
        if self.run is not None and os.path.exists(path):
            self.run.save(path)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def sample_efficiency_curve(coords, reward, args, length_scales):
    """
    Mean regret per episode for both kernels on the unviolated reward.

    The paper's claim is about sample efficiency, so a curve showing the
    invariant kernel reaching low regret in a fraction of the episodes is a
    stronger reproduction than a single tail number: it shows the mechanism
    was reproduced, not only the endpoint.

    `episodes_to_match` is the headline figure: how many episodes each kernel
    needs to first reach the worse kernel's final tail regret.
    """
    cell = choose_relevant_cell(coords, reward, args.H)

    out = {}

    for kind in ("invariant", "rbf"):
        print(f"  curve: {kind} kernel, {args.seeds} seeds", flush=True)

        runs = []

        for seed in range(args.seeds):
            print(f"    seed {seed + 1}/{args.seeds}...", flush=True)

            runs.append(
                run_one(
                    coords,
                    reward,
                    args.H,
                    args.T,
                    args.beta,
                    args.noise,
                    kind,
                    length_scales[kind],
                    seed,
                    cell,
                    args.n_fold,
                )["regret"]
            )

        n = min(len(r) for r in runs)
        stack = np.stack([np.asarray(r)[:n] for r in runs])

        out[kind] = {
            "mean": stack.mean(axis=0).tolist(),
            "se": (stack.std(axis=0, ddof=1) / np.sqrt(len(runs))).tolist()
            if len(runs) > 1
            else [float("nan")] * n,
            "cumulative": np.cumsum(stack.mean(axis=0)).tolist(),
        }

    # Target is the worse kernel's own final tail regret.
    tails = {
        k: float(np.mean(v["mean"][-max(1, len(v["mean"]) // 5) :]))
        for k, v in out.items()
    }

    target = max(tails.values())

    for k, v in out.items():
        below = np.flatnonzero(np.asarray(v["mean"]) <= target)
        v["tail_regret"] = tails[k]
        v["episodes_to_match"] = int(below[0]) if below.size else None

    out["match_target"] = target

    return out


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--side",
        type=int,
        default=5,
    )

    p.add_argument(
        "--H",
        type=int,
        default=10,
    )

    p.add_argument(
        "--T",
        type=int,
        default=200,
    )

    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--noise",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--mode",
        choices=[
            "exact",
            "relevant",
            "irrelevant",
            "flip",
        ],
        default="relevant",
        help=("Type of structural violation to test."),
    )

    p.add_argument(
        "--deltas",
        type=float,
        nargs="*",
        default=[
            0.0,
            0.05,
            0.1,
            0.2,
            0.4,
        ],
    )

    p.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--inv_len_scale",
        type=float,
        default=0.5,
        help="Repo invariant-kernel length scale.",
    )

    p.add_argument(
        "--rbf_len_scale",
        type=float,
        default=0.1,
        help="Repo RBF length scale.",
    )

    p.add_argument(
        "--n_fold",
        type=int,
        default=None,
        help=(
            "Rotation group C_n for the invariant kernel. Omit for the repo's "
            "own D4 group. 4 is a genuine symmetry of the square grid; 3, 5 "
            "and 6 are not, so the kernel pools points that differ while the "
            "environment stays exactly symmetric."
        ),
    )

    p.add_argument(
        "--curve",
        action="store_true",
        help=(
            "Sample-efficiency mode: mean regret per episode for both kernels "
            "on the unviolated reward, instead of a delta sweep."
        ),
    )

    p.add_argument(
        "--occupancy_rank",
        choices=["any", "high", "low"],
        default="any",
        help=(
            "Where to place the flip. 'high' picks among the most-visited "
            "states under the optimal policy, 'low' among the least-visited. "
            "Episodic regret averages over episodes, so a changed action only "
            "costs anything if the policy is actually there."
        ),
    )

    p.add_argument(
        "--acquisition",
        choices=list(ACQUISITIONS),
        default="ucb",
        help=(
            "Action-selection rule. 'ucb' is the repo's optimistic value "
            "iteration (the existing runs). 'ts' draws one joint posterior "
            "sample per episode-step and acts greedily on it, with a "
            "mean Bellman target. 'greedy' uses the posterior mean "
            "everywhere and is the control that isolates randomisation from "
            "the removal of the optimism bonus."
        ),
    )

    p.add_argument(
        "--kernels",
        nargs="+",
        choices=["invariant", "rbf"],
        default=["invariant", "rbf"],
        help=(
            "Which kernels to sweep. The acquisition question is about the "
            "structural prior, so 'invariant' alone halves the runtime; the "
            "RBF arm is only a no-symmetry reference."
        ),
    )

    p.add_argument(
        "--wandb_project",
        default=None,
        help=(
            "W&B project name. Omitted means no logging and W&B is stubbed "
            "out entirely, as before."
        ),
    )

    p.add_argument(
        "--wandb_entity",
        default="stephanie-jat-ucl",
    )

    p.add_argument(
        "--wandb_name",
        default=None,
        help="Run name. Defaults to the mode plus the group.",
    )

    p.add_argument(
        "--out",
        default="repo_krvi_approximate_invariance.json",
    )

    args = p.parse_args()

    print(
        "\n[1/4] Building symmetric environment...",
        flush=True,
    )

    coords = make_symmetric_grid(args.side)

    r0 = baseline_reward(coords)

    print(
        "[2/4] Checking exact D4 symmetry...",
        flush=True,
    )

    # Sanity check: the baseline is genuinely D4 invariant.
    max_symmetry_error = 0.0

    for s in range(len(coords)):
        for a in range(4):
            for si, ai in full_orbit(
                coords,
                s,
                a,
            ):
                max_symmetry_error = max(
                    max_symmetry_error,
                    abs(float(r0[s, a] - r0[si, ai])),
                )

    print(
        f"      maximum baseline symmetry error: {max_symmetry_error:.3e}",
        flush=True,
    )

    # Is the assumed group actually a symmetry of this reward?
    group_ok, group_err = group_is_valid_for(coords, r0, args.n_fold)

    kinds = tuple(dict.fromkeys(args.kernels))

    run_config = {
        "algorithm": f"repository KRVI ({args.acquisition})",
        "acquisition": args.acquisition,
        "kernels": list(kinds),
        "environment": "D4-symmetric deterministic grid",
        "violation_mode": "curve" if args.curve else args.mode,
        "side": args.side,
        "H": args.H,
        "T": args.T,
        "beta": args.beta,
        "noise": args.noise,
        "seeds": args.seeds,
        "invariant_length_scale": args.inv_len_scale,
        "rbf_length_scale": args.rbf_len_scale,
        "occupancy_rank": args.occupancy_rank,
        "n_fold": args.n_fold,
        "group": "repo_D4" if args.n_fold is None else f"C{args.n_fold}",
        "group_order": 8 if args.n_fold is None else args.n_fold,
        "group_is_valid": group_ok,
        "group_max_error": group_err,
        "baseline_symmetry_error": max_symmetry_error,
        "out": args.out,
    }

    logger = Logger(args, run_config)

    print(
        f"      assumed group: "
        f"{'repo D4' if args.n_fold is None else f'C_{args.n_fold}'}"
        f"  order={8 if args.n_fold is None else args.n_fold}"
        f"  valid={group_ok}  max error={group_err:.3e}",
        flush=True,
    )

    print(
        "[3/4] Running KRVI experiments...",
        flush=True,
    )

    # Use fixed scales throughout the sweep.
    length_scales = {
        "invariant": args.inv_len_scale,
        "rbf": args.rbf_len_scale,
    }

    if args.curve:
        curve = sample_efficiency_curve(coords, r0, args, length_scales)

        payload = {
            "algorithm": "repository KRVI",
            "environment": "D4-symmetric deterministic grid",
            "mode": "curve",
            "H": args.H,
            "T": args.T,
            "beta": args.beta,
            "noise": args.noise,
            "length_scales": length_scales,
            "occupancy_rank": args.occupancy_rank,
            "n_fold": args.n_fold,
            "group_order": 8 if args.n_fold is None else args.n_fold,
            "group_is_valid": group_ok,
            "group_max_error": group_err,
            "baseline_symmetry_error": max_symmetry_error,
            "curve": curve,
        }

        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)

        print(
            f"\n  invariant tail {curve['invariant']['tail_regret']:.4f}, "
            f"reaches target at episode "
            f"{curve['invariant']['episodes_to_match']}",
            flush=True,
        )

        print(
            f"  rbf       tail {curve['rbf']['tail_regret']:.4f}, "
            f"reaches target at episode "
            f"{curve['rbf']['episodes_to_match']}",
            flush=True,
        )

        try:
            import matplotlib

            matplotlib.use("Agg")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            for kind, style in (("invariant", "-"), ("rbf", "--")):
                m = np.asarray(curve[kind]["mean"])
                se = np.asarray(curve[kind]["se"])
                x = np.arange(len(m))

                ax[0].plot(x, m, style, label=kind)
                ax[0].fill_between(x, m - se, m + se, alpha=0.2)
                ax[1].plot(x, curve[kind]["cumulative"], style, label=kind)

            ax[0].set_xlabel("episode")
            ax[0].set_ylabel("episodic regret")
            ax[0].set_title("sample efficiency at exact symmetry")
            ax[0].legend()

            ax[1].set_xlabel("episode")
            ax[1].set_ylabel("cumulative regret")
            ax[1].set_title("cumulative regret")
            ax[1].legend()

            fig.tight_layout()
            fig.savefig(args.out.replace(".json", ".png"), dpi=150)

            print(f"  written {args.out.replace('.json', '.png')}", flush=True)

        except Exception as exc:
            print(f"  (plot skipped: {exc})", flush=True)

        # Per-episode series so W&B renders a curve, not a scatter.
        for i in range(len(curve["invariant"]["mean"])):
            logger.log(
                {
                    "curve/regret_invariant": curve["invariant"]["mean"][i],
                    "curve/regret_rbf": curve["rbf"]["mean"][i],
                    "curve/se_invariant": curve["invariant"]["se"][i],
                    "curve/se_rbf": curve["rbf"]["se"][i],
                    "curve/cum_invariant": curve["invariant"]["cumulative"][i],
                    "curve/cum_rbf": curve["rbf"]["cumulative"][i],
                },
                step=i,
            )

        for kind in ("invariant", "rbf"):
            logger.summary(f"tail_regret/{kind}", curve[kind]["tail_regret"])
            logger.summary(
                f"episodes_to_match/{kind}", curve[kind]["episodes_to_match"]
            )

        logger.summary("match_target", curve["match_target"])
        logger.save(args.out)
        logger.save(args.out.replace(".json", ".png"))
        logger.finish()

        print(f"  written {args.out}", flush=True)

        return

    rows = []
    trajectories = {}

    # -----------------------------------------------------------------------
    # Determine the delta sweep.
    # -----------------------------------------------------------------------

    flip_cell = None
    flip_margin = None
    flip_best_action = None
    flip_occupancy = None
    deltas = list(args.deltas)

    if args.mode == "flip":
        (
            flip_cell,
            flip_margin,
            flip_best_action,
            flip_occupancy,
        ) = choose_flip_cell(
            coords,
            r0,
            args.H,
            occupancy_rank=args.occupancy_rank,
        )

        # Sweep relative to the actual baseline action margin so the
        # decision boundary is deliberately bracketed.
        deltas = [
            0.0,
            0.25 * flip_margin,
            0.5 * flip_margin,
            0.9 * flip_margin,
            1.1 * flip_margin,
            1.5 * flip_margin,
            2.0 * flip_margin,
        ]

        print(
            f"\nflip mode:"
            f"\n  state = {flip_cell[0]}"
            f"\n  perturbed action = {flip_cell[1]}"
            f"\n  baseline optimal action = {flip_best_action}"
            f"\n  occupancy rank = {args.occupancy_rank}"
            f"\n  state visit frequency = {flip_occupancy:.4f}"
            f"\n  baseline margin = {flip_margin:.6f}"
            f"\n  predicted crossing = delta ~= {flip_margin:.6f}"
            f"\n  bracket = "
            f"{0.9 * flip_margin:.6f} -> "
            f"{1.1 * flip_margin:.6f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Delta sweep.
    # -----------------------------------------------------------------------

    for delta_idx, delta in enumerate(
        deltas,
        start=1,
    ):
        print(
            f"\n  delta = {delta:.6f} ({delta_idx}/{len(deltas)})",
            flush=True,
        )

        rec = {
            "delta": float(delta),
            "regret": {},
            "regret_se": {},
            "orbit_gap": {},
            "lml": {},
        }

        # Use one fixed violation realization per delta/mode.
        r_v, cell = violate(
            coords,
            r0,
            delta,
            args.mode,
            args.H,
            flip_cell=flip_cell,
        )

        # ---------------------------------------------------------------
        # True reward gap across the 180-degree orbit.
        # ---------------------------------------------------------------

        partner = (
            rotate180_state_partner(
                coords,
                cell[0],
            ),
            opposite_action(cell[1]),
        )

        true_reward_gap = float(abs(r_v[cell] - r_v[partner]))

        # ---------------------------------------------------------------
        # Check whether the optimal action actually changes.
        # ---------------------------------------------------------------

        _, Q_before = optimal_values(
            coords,
            r0,
            args.H,
        )

        _, Q_after = optimal_values(
            coords,
            r_v,
            args.H,
        )

        before_action = int(np.argmax(Q_before[cell[0]]))

        after_action = int(np.argmax(Q_after[cell[0]]))

        rec["optimal_action_before"] = before_action

        rec["optimal_action_after"] = after_action

        rec["decision_changed"] = bool(before_action != after_action)

        rec["violation_cell"] = list(cell)

        rec["orbit_partner"] = list(partner)

        rec["true_reward_gap"] = true_reward_gap

        # ---------------------------------------------------------------
        # Flip-specific metadata.
        # ---------------------------------------------------------------

        if args.mode == "flip":
            rec["flip_margin"] = float(flip_margin)

            rec["flip_state_occupancy"] = float(flip_occupancy)

            rec["occupancy_rank"] = args.occupancy_rank

            rec["flip_baseline_optimal_action"] = int(flip_best_action)

            rec["flip_predicted_crossing_delta"] = float(flip_margin)

            rec["flip_relative_to_margin"] = (
                float(delta / flip_margin) if flip_margin > 0 else None
            )

        # ---------------------------------------------------------------
        # Run both kernels.
        # ---------------------------------------------------------------

        for kind in kinds:
            print(
                f"    {kind} kernel, {args.acquisition}: {args.seeds} seeds",
                flush=True,
            )

            regrets = []
            gaps = []
            lml_inv = []
            lml_rbf = []
            decomps = []
            decomps_full = []
            residuals = []
            fallbacks = 0

            for seed in range(args.seeds):
                print(
                    f"      seed {seed + 1}/{args.seeds}...",
                    flush=True,
                )

                out = run_one(
                    coords,
                    r_v,
                    args.H,
                    args.T,
                    args.beta,
                    args.noise,
                    kind,
                    length_scales[kind],
                    seed,
                    cell,
                    args.n_fold,
                    acquisition=args.acquisition,
                )

                residuals.append(out["orbit_residual"])
                fallbacks += out["ts_fallbacks"]

                # Tail regret: last fifth of the recorded episodes.
                tail_n = max(
                    1,
                    args.T // 5,
                )

                if len(out["regret"]) > 0:
                    regrets.append(float(out["regret"][-tail_n:].mean()))

                if out["orbit_gap"] is not None:
                    gaps.append(float(out["orbit_gap"]))

                decomps.append(
                    decompose_regret(
                        coords,
                        r_v,
                        args.H,
                        out["episodes"],
                        cell,
                        tail_n,
                    )
                )

                # Same accounting over EVERY episode, not just the converged
                # tail. Exploration happens early, so whether an acquisition
                # rule ever covered the violated orbit is a whole-run
                # question; tail_n=0 means "all episodes".
                decomps_full.append(
                    decompose_regret(
                        coords,
                        r_v,
                        args.H,
                        out["episodes"],
                        cell,
                        0,
                    )
                )

                lml_inv.append(out["lml"]["invariant"])

                lml_rbf.append(out["lml"]["rbf"])

                trajectories.setdefault(
                    (
                        kind,
                        seed,
                    ),
                    [],
                ).append(
                    (
                        float(delta),
                        out["trajectory"],
                    )
                )

            rec["regret"][kind] = float(np.mean(regrets))

            rec["regret_se"][kind] = stderr(regrets)

            rec["orbit_gap"][kind] = float(np.mean(gaps)) if gaps else None

            decomp_summary = {}

            for key in decomps[0]:
                vals = [d[key] for d in decomps if np.isfinite(d[key])]
                decomp_summary[key] = float(np.mean(vals)) if vals else None
                decomp_summary[key + "_se"] = stderr(vals) if len(vals) > 1 else None

            rec.setdefault("decomposition", {})[kind] = decomp_summary

            full_summary = {}

            for key in decomps_full[0]:
                vals = [d[key] for d in decomps_full if np.isfinite(d[key])]
                full_summary[key] = float(np.mean(vals)) if vals else None
                full_summary[key + "_se"] = stderr(vals) if len(vals) > 1 else None

            rec.setdefault("decomposition_full", {})[kind] = full_summary

            # Orbit residual contrast, averaged over seeds. Seeds that never
            # reached one side of the orbit contribute no z; the count of
            # usable seeds is reported so an absent z is distinguishable from
            # a z of zero.
            resid_summary = {}

            for key in (
                "n_rows_violated_state",
                "n_rows_orbit_states",
                "mean_residual_violated_state",
                "mean_residual_orbit_states",
                "contrast",
                "z",
            ):
                vals = [
                    r[key]
                    for r in residuals
                    if r.get(key) is not None and np.isfinite(r[key])
                ]
                resid_summary[key] = float(np.mean(vals)) if vals else None
                resid_summary[key + "_se"] = stderr(vals) if len(vals) > 1 else None
                resid_summary[key + "_n_seeds"] = len(vals)

            resid_summary["ts_fallbacks"] = fallbacks

            rec.setdefault("orbit_residual", {})[kind] = resid_summary

            if fallbacks:
                print(
                    f"      WARNING: {fallbacks} joint posterior draws fell "
                    f"back to independent marginal sampling; this arm is no "
                    f"longer Thompson sampling and should not be reported",
                    flush=True,
                )

            if (
                decomp_summary["telescoping_error"]
                and decomp_summary["telescoping_error"] > 1e-6
            ):
                print(
                    f"      WARNING: regret decomposition does not add up "
                    f"(max mismatch {decomp_summary['telescoping_error']:.2e}); "
                    f"the state reconstruction disagrees with the environment",
                    flush=True,
                )

            # These are evaluated on the same Bellman data within each run.
            rec["lml"][f"invariant@{length_scales[kind]:g}"] = float(np.mean(lml_inv))

            rec["lml"][f"rbf@{length_scales[kind]:g}"] = float(np.mean(lml_rbf))

        # ---------------------------------------------------------------
        # Same-data LML ratios.
        # ---------------------------------------------------------------

        common_scale_pairs = [
            (
                args.inv_len_scale,
                f"invariant@{args.inv_len_scale:g}",
                f"rbf@{args.inv_len_scale:g}",
            ),
            (
                args.rbf_len_scale,
                f"invariant@{args.rbf_len_scale:g}",
                f"rbf@{args.rbf_len_scale:g}",
            ),
        ]

        rec["lml_ratio"] = {}

        for (
            scale,
            ikey,
            rkey,
        ) in common_scale_pairs:
            if ikey in rec["lml"] and rkey in rec["lml"]:
                rec["lml_ratio"][f"{scale:g}"] = rec["lml"][ikey] - rec["lml"][rkey]

        rows.append(rec)

        decomposition = rec.get("decomposition", {})

        def fmt(value, digits=3):
            # leak_rate / fix_rate are None when the orbit (or the violated
            # state) was never visited in the window. That is a result, not a
            # missing value, and it is exactly what a non-optimistic
            # acquisition rule is expected to produce -- so it must print,
            # not raise.
            return "n/a" if value is None else f"{value:.{digits}f}"

        if decomposition:
            for kind in kinds:
                d = decomposition.get(kind, {})
                print(
                    f"      {kind:9s} loss at violated state {fmt(d.get('loss_violated'))}"
                    f"  at mirror states {fmt(d.get('loss_orbit'))}"
                    f"  elsewhere {fmt(d.get('loss_rest'))}"
                    f"  | mirror-action rate {fmt(d.get('leak_rate'), 2)}",
                    flush=True,
                )

        # Live per-delta logging, in the same shape as the repo's train loop:
        # one W&B step per sweep point, flat keys so panels group cleanly.
        metrics = {
            "delta": float(delta),
            "true_reward_gap": rec["true_reward_gap"],
            "decision_changed": int(rec["decision_changed"]),
            "optimal_action_before": rec["optimal_action_before"],
            "optimal_action_after": rec["optimal_action_after"],
        }

        for kind in kinds:
            metrics[f"regret/{kind}"] = rec["regret"][kind]
            metrics[f"regret_se/{kind}"] = rec["regret_se"][kind]
            metrics[f"orbit_gap/{kind}"] = rec["orbit_gap"][kind]

        # decomp/   tail-window accounting (as before)
        # full/     the same accounting over every episode
        # resid/    orbit residual contrast
        for prefix, block in (
            ("decomp", "decomposition"),
            ("full", "decomposition_full"),
            ("resid", "orbit_residual"),
        ):
            for kind, summary in rec.get(block, {}).items():
                for key, value in summary.items():
                    if value is not None and not key.endswith("_se"):
                        metrics[f"{prefix}/{key}/{kind}"] = value

        for key, value in rec["lml"].items():
            metrics[f"lml/{key}"] = value

        for key, value in rec["lml_ratio"].items():
            metrics[f"lml_ratio/{key}"] = value

        if args.mode == "flip":
            metrics["flip/relative_to_margin"] = rec["flip_relative_to_margin"]

        logger.log(metrics, step=delta_idx - 1)

        summary_bits = []

        for kind in kinds:
            resid = rec.get("orbit_residual", {}).get(kind, {})

            z = resid.get("z")

            full = rec["decomposition_full"][kind]

            gap = rec["orbit_gap"][kind]
            gap_text = "n/a" if gap is None else f"{gap:.3e}"

            summary_bits.append(
                f"{kind}: regret={fmt(rec['regret'][kind], 4)}"
                f"+/-{fmt(rec['regret_se'][kind], 4)}"
                f" orbit-gap={gap_text}"
                f" pair-visits(violated,orbit)="
                f"{fmt(full['violated_pair_visits'], 1)},"
                f"{fmt(full['orbit_pair_visits'], 1)}"
                f" resid-z={fmt(z, 2)}"
            )

        print(
            f"delta={delta:8.6f}  true-reward-gap={true_reward_gap:.4f}  "
            f"decision-changed={rec['decision_changed']}\n    "
            + "\n    ".join(summary_bits),
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Behaviour-change diagnostic.
    # -----------------------------------------------------------------------

    if len(deltas) > 1:
        identical = 0
        total = 0

        baseline_delta = float(deltas[0])

        for seq in trajectories.values():
            base = dict(seq).get(baseline_delta)

            for d, traj in seq:
                if d == baseline_delta:
                    continue

                total += 1
                identical += int(traj == base)

        print(
            f"\ntrajectory identical to "
            f"delta={baseline_delta:.6f} "
            f"in {identical}/{total} "
            f"kernel-seed comparisons"
        )

    # -----------------------------------------------------------------------
    # Write results.
    # -----------------------------------------------------------------------

    print(
        "\n[4/4] Writing results and plots...",
        flush=True,
    )

    output = {
        "algorithm": "repository KRVI",
        "environment": "D4-symmetric deterministic grid",
        "mode": args.mode,
        "H": args.H,
        "T": args.T,
        "beta": args.beta,
        "noise": args.noise,
        "length_scales": length_scales,
        "baseline_symmetry_error": max_symmetry_error,
        "rows": rows,
    }

    if args.mode == "flip":
        output["flip"] = {
            "cell": list(flip_cell),
            "margin": float(flip_margin),
            "baseline_optimal_action": int(flip_best_action),
            "predicted_crossing_delta": float(flip_margin),
            "state_occupancy": float(flip_occupancy),
            "occupancy_rank": args.occupancy_rank,
            "sweep": deltas,
        }

    with open(
        args.out,
        "w",
    ) as fh:
        json.dump(
            output,
            fh,
            indent=2,
        )

    print(f"written {args.out}")

    # Summary quantities: the crossings are what the proposal quotes.
    crossing = next(
        (r["delta"] for r in rows if r.get("decision_changed")),
        None,
    )

    logger.summary("decision_change_delta", crossing)

    # Only defined when both kernels were swept.
    if {"invariant", "rbf"} <= set(kinds):
        logger.summary(
            "regret_crossover_delta",
            next(
                (
                    r["delta"]
                    for r in rows
                    if r["regret"]["invariant"] > r["regret"]["rbf"]
                ),
                None,
            ),
        )

    if args.mode == "flip":
        logger.summary("flip_margin", float(flip_margin))
        logger.summary("flip_state_occupancy", float(flip_occupancy))

    # -----------------------------------------------------------------------
    # Plots.
    # -----------------------------------------------------------------------

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        d = [x["delta"] for x in rows]

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(14, 4),
        )

        # ---------------------------------------------------------------
        # Panel 1: regret.
        # ---------------------------------------------------------------

        for (
            kind,
            marker,
            label,
        ) in (
            (
                "invariant",
                "o-",
                "repo invariant kernel",
            ),
            (
                "rbf",
                "s-",
                "repo RBF",
            ),
        ):
            ax[0].errorbar(
                d,
                [x["regret"][kind] for x in rows],
                yerr=[x["regret_se"][kind] for x in rows],
                fmt=marker,
                capsize=3,
                label=label,
            )

        ax[0].set_xlabel("reward violation δ")

        ax[0].set_ylabel("tail episodic regret")

        ax[0].set_title("KRVI: cost of structural pooling")

        ax[0].legend()

        # ---------------------------------------------------------------
        # Panel 2: model orbit gap vs true reward gap.
        # ---------------------------------------------------------------

        for (
            kind,
            marker,
        ) in (
            (
                "invariant",
                "o-",
            ),
            (
                "rbf",
                "s-",
            ),
        ):
            ax[1].plot(
                d,
                [x["orbit_gap"][kind] for x in rows],
                marker,
                label=kind,
            )

        ax[1].plot(
            d,
            [x["true_reward_gap"] for x in rows],
            "k--",
            label="true reward gap",
        )

        ax[1].set_xlabel("reward violation δ")

        ax[1].set_ylabel("gap across 180° orbit")

        ax[1].set_title("can the model represent the distinction?")

        ax[1].legend()

        # ---------------------------------------------------------------
        # Panel 3: relative model evidence.
        # ---------------------------------------------------------------

        for scale in (
            args.inv_len_scale,
            args.rbf_len_scale,
        ):
            key = f"{scale:g}"

            vals = [
                x["lml_ratio"].get(
                    key,
                    np.nan,
                )
                for x in rows
            ]

            ax[2].plot(
                d,
                vals,
                "o-",
                label=f"scale {key}",
            )

        ax[2].axhline(
            0.0,
            color="k",
            lw=0.8,
        )

        ax[2].set_xlabel("reward violation δ")

        ax[2].set_ylabel("LML(invariant) − LML(RBF)")

        ax[2].set_title("relative model evidence")

        ax[2].legend()

        fig.tight_layout()

        png = args.out.replace(
            ".json",
            ".png",
        )

        fig.savefig(
            png,
            dpi=150,
        )

        plt.close(fig)

        print(f"written {png}")

    except Exception as exc:
        print(f"(plot skipped: {exc})")

    # Where the regret was paid. Separate figure so the original is unchanged.
    decomposition_png = args.out.replace(".json", "_decomposition.png")

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        have = [r for r in rows if r.get("decomposition")]

        if have:
            d = [r["delta"] for r in have]

            fig, ax = plt.subplots(1, 3, figsize=(14, 4))

            panels = (
                ("loss_violated", "regret paid at the violated state"),
                (
                    "loss_orbit",
                    "regret paid at its mirror states\n(environment unchanged there)",
                ),
                (
                    "leak_rate",
                    "how often the mirrored action\nis taken at mirror states",
                ),
            )

            for axis, (key, title) in zip(ax, panels):
                for kind, marker, label in (
                    ("invariant", "o-", "invariant kernel"),
                    ("rbf", "s-", "plain RBF"),
                ):
                    y = [r["decomposition"][kind][key] for r in have]
                    se = [
                        r["decomposition"][kind].get(key + "_se") or 0.0 for r in have
                    ]
                    axis.errorbar(d, y, yerr=se, fmt=marker, capsize=3, label=label)

                if args.mode == "flip" and flip_margin is not None:
                    axis.axvline(flip_margin, color="k", lw=0.8, ls=":")

                axis.set_xlabel("reward violation δ")
                axis.set_title(title)
                axis.legend()

            ax[0].set_ylabel("regret per episode (tail)")
            ax[2].set_ylabel("fraction of visits")

            fig.tight_layout()
            fig.savefig(decomposition_png, dpi=150)
            plt.close(fig)

            print(f"written {decomposition_png}")

    except Exception as exc:
        print(f"(decomposition plot skipped: {exc})")

    # Attach the JSON and PNGs to the run, then close it. All files are written
    # regardless of whether W&B is enabled.
    logger.save(args.out)
    logger.save(args.out.replace(".json", ".png"))
    logger.save(decomposition_png)
    logger.finish()


if __name__ == "__main__":
    main()
