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

import KRVI_algo_test_rotated_invariant_fulloptim as repo
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

# ---------------------------------------------------------------------------
# Import up-to-date W&B only when the user asks for logging, because the repo
# calls wandb.Settings(start_method="thread"), which newer versions reject.
# This file also does its own logging.
# ---------------------------------------------------------------------------


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
# Import actual repo implementation.
# ---------------------------------------------------------------------------


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise RuntimeError(
        "This script needs gymnasium because the repo's KRVI expects a "
        "gym-style environment."
    ) from exc


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


def choose_flip_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
    min_margin: float = 1e-6,
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

    candidates = []

    for s in range(len(coords)):
        order = np.argsort(Q0[s])

        best_a = int(order[-1])
        runner_a = int(order[-2])

        margin = float(Q0[s, best_a] - Q0[s, runner_a])

        if margin <= min_margin:
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
            "No state with a breakable positive action margin was found."
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


class ProbedKRVI(repo.KRVI):
    """
    Thin subclass: the learning algorithm remains the repo's KRVI.

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
):
    """
    Instantiate KRVI while passing only constructor arguments that the local
    copy of the repo actually accepts.
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

    return ProbedKRVI(**accepted)


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

    run_config = {
        "algorithm": "repository KRVI",
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
    deltas = list(args.deltas)

    if args.mode == "flip":
        (
            flip_cell,
            flip_margin,
            flip_best_action,
        ) = choose_flip_cell(
            coords,
            r0,
            args.H,
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

            rec["flip_baseline_optimal_action"] = int(flip_best_action)

            rec["flip_predicted_crossing_delta"] = float(flip_margin)

            rec["flip_relative_to_margin"] = (
                float(delta / flip_margin) if flip_margin > 0 else None
            )

        # ---------------------------------------------------------------
        # Run both kernels.
        # ---------------------------------------------------------------

        for kind in (
            "invariant",
            "rbf",
        ):
            print(
                f"    {kind} kernel: {args.seeds} seeds",
                flush=True,
            )

            regrets = []
            gaps = []
            lml_inv = []
            lml_rbf = []

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
                )

                # Tail regret: last fifth of the recorded episodes.
                tail_n = max(
                    1,
                    args.T // 5,
                )

                if len(out["regret"]) > 0:
                    regrets.append(float(out["regret"][-tail_n:].mean()))

                if out["orbit_gap"] is not None:
                    gaps.append(float(out["orbit_gap"]))

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

        # Live per-delta logging, in the same shape as the repo's train loop:
        # one W&B step per sweep point, flat keys so panels group cleanly.
        metrics = {
            "delta": float(delta),
            "regret/invariant": rec["regret"]["invariant"],
            "regret/rbf": rec["regret"]["rbf"],
            "regret_se/invariant": rec["regret_se"]["invariant"],
            "regret_se/rbf": rec["regret_se"]["rbf"],
            "orbit_gap/invariant": rec["orbit_gap"]["invariant"],
            "orbit_gap/rbf": rec["orbit_gap"]["rbf"],
            "true_reward_gap": rec["true_reward_gap"],
            "decision_changed": int(rec["decision_changed"]),
            "optimal_action_before": rec["optimal_action_before"],
            "optimal_action_after": rec["optimal_action_after"],
        }

        for key, value in rec["lml"].items():
            metrics[f"lml/{key}"] = value

        for key, value in rec["lml_ratio"].items():
            metrics[f"lml_ratio/{key}"] = value

        if args.mode == "flip":
            metrics["flip/relative_to_margin"] = rec["flip_relative_to_margin"]

        logger.log(metrics, step=delta_idx - 1)

        print(
            f"delta={delta:8.6f}  "
            f"regret inv="
            f"{rec['regret']['invariant']:.4f}"
            f"+/-"
            f"{rec['regret_se']['invariant']:.4f}  "
            f"rbf="
            f"{rec['regret']['rbf']:.4f}"
            f"+/-"
            f"{rec['regret_se']['rbf']:.4f}  "
            f"orbit-gap inv="
            f"{rec['orbit_gap']['invariant']:.4e} "
            f"rbf="
            f"{rec['orbit_gap']['rbf']:.4f}  "
            f"true-reward-gap="
            f"{true_reward_gap:.4f}  "
            f"decision-changed="
            f"{rec['decision_changed']}",
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

    logger.summary(
        "regret_crossover_delta",
        next(
            (r["delta"] for r in rows if r["regret"]["invariant"] > r["regret"]["rbf"]),
            None,
        ),
    )

    if args.mode == "flip":
        logger.summary("flip_margin", float(flip_margin))

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

    # Attach the JSON and PNG to the run, then close it. Both files are written
    # regardless of whether W&B is enabled.
    logger.save(args.out)
    logger.save(args.out.replace(".json", ".png"))
    logger.finish()


if __name__ == "__main__":
    main()
