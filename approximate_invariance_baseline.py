#!/usr/bin/env python3
"""
Controlled approximate-invariance experiment using the repo-native KRVI implementation.
The purpose of this file is to demonstrate the baseline behaviour: 
1. the KRVI algorithm behaves differently with different kernels
    - with oracle invariant kernel (delta=0), Q-gap ~0;
    - with RBF, Q-gap ~1.27;
2. cost increases as the invariance is violated:
    - invariant regret grows from ~0.074 to ~0.303, proportionate to size of delta violation. 

This file:
  1. imports KRVI, InvariantKernel and RBFKernel from
     KRVI_algo_test_rotated_invariant_fulloptim.py;
  2. gives that exact KRVI implementation a small D4-symmetric grid environment;
  3. perturbs one state-action reward by delta (or spreads perturbations across
     orbits);
  4. captures the quantities needed for the proposal:
       - tail episodic regret,
       - whether behaviour changes,
       - the model's predicted Q-gap across an orbit,
       - log-marginal-likelihood ratio on the same Bellman data;
  5. leaves the actual KRVI GP/value-iteration/training machinery to the repo.

A custom environment is built because the repo's original Frozen Lake example 
is not itself D4-symmetric, so it cannot give a clean delta=0 exact-invariance baseline. 
The environment introduced here is symmetric by construction, 
while the learner is still the repo's KRVI.

Place in repo root next to KRVI file, then run baseline experiment:

    python3 approximate_invariance_baseline.py \
        --H 10 --T 200 --seeds 5 --beta 0.1 \
        --deltas 0 \
        --out baseline.json

Option to vary delta:

    python3 approximate_invariance_baseline.py \
        --H 10 --T 200 --seeds 5 --beta 0.1 \
        --deltas 0 0.05 0.1 0.2 0.4 \
        --out point_violation.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import json
import sys
import types
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# The repo currently calls wandb.Settings(start_method="thread"), which is
# incompatible with newer W&B versions. We do not need logging for this
# experiment, so W&B is stubbed BEFORE importing the repo module.
# ---------------------------------------------------------------------------

if "wandb" not in sys.modules:
    stub = types.ModuleType("wandb")
    stub.init = lambda *a, **k: None
    stub.log = lambda *a, **k: None
    stub.Image = lambda *a, **k: None
    stub.run = types.SimpleNamespace(summary={})
    sys.modules["wandb"] = stub


# ---------------------------------------------------------------------------
# Import the actual repo implementation
# ---------------------------------------------------------------------------

import KRVI_algo_test_rotated_invariant_fulloptim as repo

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

    The reward is supplied as a state-action matrix.  The baseline reward is
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
        return self.coords[next_state].copy(), reward, False, False, {}


def len_side(coords: np.ndarray) -> int:
    """Number of points on one coordinate axis."""
    return int(round(np.sqrt(len(coords))))


def nearest_state_index(coords: np.ndarray, point: np.ndarray) -> int:
    d2 = np.sum((coords - point.reshape(1, 2)) ** 2, axis=1)
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

    Reward is highest near the origin.  It is intentionally simple:
    the experiment is about the learner's structural assumption,
    not about making a complicated benchmark.
    """
    # State-only reward, replicated over the four actions.
    dist2 = np.sum(coords**2, axis=1)
    state_reward = 1.0 - dist2
    return np.repeat(state_reward[:, None], 4, axis=1)


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

    # Work backwards. Q_h uses V_{h+1}.
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
# Controlled violations
# ---------------------------------------------------------------------------


def rotate180_state_partner(
    coords: np.ndarray,
    state_idx: int,
) -> int:
    target = -coords[state_idx]
    return nearest_state_index(coords, target)


def opposite_action(action_idx: int) -> int:
    return {0: 2, 1: 3, 2: 0, 3: 1}[int(action_idx)]


def full_orbit(
    coords: np.ndarray,
    state_idx: int,
    action_idx: int,
):
    """
    Return the D4 orbit generated by the repo's rotation group.

    The repo's invariant kernel treats simultaneous rotations of the state and
    action coordinates as equivalent.  For diagnostics we only need the
    orbit partner reached by 180 degrees, which is enough to expose a
    non-zero structural distinction.
    """
    x, y = coords[state_idx]
    a = ACTIONS[action_idx]

    orbit = []
    for k in range(4):
        theta = k * np.pi / 2.0
        c, s = np.cos(theta), np.sin(theta)

        R = np.array([[c, -s], [s, c]])
        sx = R @ np.array([x, y])
        av = R @ a

        si = nearest_state_index(coords, sx)
        ai = int(np.argmin(np.sum((ACTIONS - av.reshape(1, 2)) ** 2, axis=1)))
        orbit.append((si, ai))

    # Preserve order but remove duplicates.
    return list(dict.fromkeys(orbit))


def choose_violation_cell(
    coords: np.ndarray,
    reward: np.ndarray,
    horizon: int,
):
    """
    Put the point violation at a genuinely decision-relevant state-action:
    the baseline optimal Q_0 maximum, rather than merely the largest reward.
    """
    _, Q0 = optimal_values(coords, reward, horizon)
    s, a = np.unravel_index(int(np.argmax(Q0)), Q0.shape)
    return int(s), int(a)


def violate(
    coords: np.ndarray,
    reward: np.ndarray,
    delta: float,
    mode: str,
    rng: np.random.Generator,
    horizon: int,
):
    out = reward.copy()

    anchor_s, anchor_a = choose_violation_cell(coords, reward, horizon)

    if mode == "point":
        # One decision-relevant member of an orbit is changed.
        out[anchor_s, anchor_a] += delta
        return out, (anchor_s, anchor_a)

    # Spread mode: one representative per orbit, with a fixed random sign/
    # magnitude.  The same xi is reused for every delta.
    xi = rng.standard_normal(out.shape)
    xi /= max(np.max(np.abs(xi)), 1e-12)

    seen = set()
    for s in range(len(coords)):
        for a in range(4):
            orb = full_orbit(coords, s, a)
            key = tuple(sorted(orb))
            if key in seen:
                continue
            seen.add(key)
            # Use the first representative only.
            si, ai = orb[0]
            out[si, ai] += delta * xi[si, ai]

    return out, (anchor_s, anchor_a)


# ---------------------------------------------------------------------------
# Repo KRVI probe
# ---------------------------------------------------------------------------


class ProbedKRVI(repo.KRVI):
    """
    Thin subclass: learning algorithm is KRVI.

    The only addition is recording the GP/data produced by
    GP_regression_torch so we can calculate diagnostics after train().
    """

    def __init__(self, *args, **kwargs):
        self.gp_calls = []
        self.last_gp = None
        self.last_X = None
        self.last_y = None
        super().__init__(*args, **kwargs)

    def GP_regression_torch(self, X, y):
        gp = super().GP_regression_torch(X, y)

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


def make_repo_kernel(kind: str):
    if kind == "invariant":
        return repo.InvariantKernel(
            base_kernel=repo.RBFKernel(),
            transformations=repo.apply_rotation_group,
            is_isotropic=True,
            is_group=True,
        )
    if kind == "rbf":
        return repo.RBFKernel()
    raise ValueError(kind)


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

    This makes the file robust to small signature changes in the repo.
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
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        return float(mll(output, gp.train_targets).item())


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
) -> float:
    """
    Fit the repo's GP implementation with the other kernel on exactly the
    same Bellman regression data.

    This is the passive-data model comparison: the data are held fixed while
    only the structural kernel changes.
    """
    probe = make_repo_krvi(
        make_repo_kernel(kind),
        env,
        beta,
        horizon,
        length_scale,
        noise,
        seed,
    )
    gp = probe.GP_regression_torch(X, y)
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

    states = np.vstack([state, partner_state])
    actions = np.vstack([action, partner_action])

    mean, _, _ = probe.predict_with_gp(probe.last_gp, states, actions)
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
):
    env = SymmetricGridEnv(
        reward=reward,
        coords=coords,
        rng=np.random.default_rng(seed),
    )

    probe = make_repo_krvi(
        make_repo_kernel(kind),
        env,
        beta,
        horizon,
        length_scale,
        noise,
        seed,
    )

    # The actual repo KRVI training loop.
    probe.train(T)

    # The logging wrapper is deliberately simple: because KRVI talks directly
    # to env.reset()/env.step(), we collect trajectories by running the same
    # env interface through a second lightweight recording wrapper below.
    #
    # Instead of trying to infer trajectories from KRVI's internal locals,
    # replaying is NOT used here.  The environment records them during train.
    #
    # To make that possible, SymmetricGridEnv stores them:
    # (see the monkey-patch immediately below).
    episodes = getattr(env, "episodes", [])

    # Repo's final h=0 regression model.
    orbit_gap = predict_orbit_gap(probe, coords, run_one.cell)

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
        )
        if probe.last_X is not None
        else None
    )

    # Exact finite-horizon optimal value for the same violated reward.
    V_star, _ = optimal_values(coords, reward, horizon)

    regrets = []
    trajectories = []

    for ep in episodes:
        start = ep["start"]
        total_reward = sum(ep["rewards"])
        regrets.append(float(V_star[start] - total_reward))
        trajectories.append(tuple(ep["actions"]))

    return {
        "regret": np.asarray(regrets, dtype=float),
        "trajectory": tuple(trajectories),
        "orbit_gap": orbit_gap,
        "lml": {
            "invariant": lml_same_kernel if kind == "invariant" else other_lml,
            "rbf": lml_same_kernel if kind == "rbf" else other_lml,
        },
        "last_X": probe.last_X,
        "last_y": probe.last_y,
    }


# ---------------------------------------------------------------------------
# Add recording to the environment without touching KRVI.
# ---------------------------------------------------------------------------

_original_reset = SymmetricGridEnv.reset
_original_step = SymmetricGridEnv.step


def _recording_reset(self, *args, **kwargs):
    result = _original_reset(self, *args, **kwargs)
    self._episode_start = int(self._state)
    self._episode_actions = []
    self._episode_rewards = []
    if not hasattr(self, "episodes"):
        self.episodes = []
    return result


def _recording_step(self, action):
    start_state = self._state
    result = _original_step(self, action)
    _, reward, _, _, _ = result

    self._episode_actions.append(int(action))
    self._episode_rewards.append(float(reward))

    # The repo's KRVI calls reset() at the start of each episode, so once the
    # next reset happens the previous episode is complete.  We cannot append
    # here based on terminal because this environment is intentionally
    # non-terminal.
    self._current_episode_snapshot = {
        "start": self._episode_start,
        "actions": list(self._episode_actions),
        "rewards": list(self._episode_rewards),
    }
    return result


def _recording_reset_and_archive(self, *args, **kwargs):
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

    result = _original_reset(self, *args, **kwargs)
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
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, default=5)
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--mode", choices=["point", "spread"], default="point")
    p.add_argument(
        "--deltas",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.2, 0.4],
    )
    p.add_argument("--seeds", type=int, default=5)
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
        "--out",
        default="repo_krvi_approximate_invariance.json",
    )
    args = p.parse_args()

    print("\n[1/4] Building symmetric environment...", flush=True)
    coords = make_symmetric_grid(args.side)
    r0 = baseline_reward(coords)

    print("[2/4] Checking exact D4 symmetry...", flush=True)

    # Sanity check: the baseline is genuinely D4 invariant.
    max_symmetry_error = 0.0
    for s in range(len(coords)):
        for a in range(4):
            for si, ai in full_orbit(coords, s, a):
                max_symmetry_error = max(
                    max_symmetry_error,
                    abs(float(r0[s, a] - r0[si, ai])),
                )

    print(
        f"      maximum baseline symmetry error: {max_symmetry_error:.3e}",
        flush=True,
    )
    print("[3/4] Running KRVI experiments...", flush=True)

    # Use the same fixed scales throughout the sweep.  This avoids the
    # earlier tuning confound where each kernel was selected by a short-run
    # regret criterion.
    length_scales = {
        "invariant": args.inv_len_scale,
        "rbf": args.rbf_len_scale,
    }

    rows = []
    trajectories = {}

    for delta_idx, delta in enumerate(args.deltas, start=1):
        print(
            f"\n  delta = {delta:.3f} ({delta_idx}/{len(args.deltas)})",
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
            np.random.default_rng(0),
            args.H,
        )
        run_one.cell = cell

        # True reward gap across the 180-degree orbit is diagnostic only.
        partner = (
            rotate180_state_partner(coords, cell[0]),
            opposite_action(cell[1]),
        )
        true_reward_gap = float(abs(r_v[cell] - r_v[partner]))

        rec["violation_cell"] = list(cell)
        rec["orbit_partner"] = list(partner)
        rec["true_reward_gap"] = true_reward_gap

        for kind in ("invariant", "rbf"):
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
                # Fresh environment/reward object for each seed.
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
                )

                regrets.append(float(out["regret"][-max(1, args.T // 5) :].mean()))
                if out["orbit_gap"] is not None:
                    gaps.append(float(out["orbit_gap"]))

                lml_inv.append(out["lml"]["invariant"])
                lml_rbf.append(out["lml"]["rbf"])

                trajectories.setdefault((kind, seed), []).append(
                    (float(delta), out["trajectory"])
                )

            rec["regret"][kind] = float(np.mean(regrets))
            rec["regret_se"][kind] = stderr(regrets)
            rec["orbit_gap"][kind] = float(np.mean(gaps)) if gaps else None

            # These are already evaluated on the same Bellman data within
            # each run. Average them across seeds.
            rec["lml"][f"invariant@{length_scales[kind]:g}"] = float(np.mean(lml_inv))
            rec["lml"][f"rbf@{length_scales[kind]:g}"] = float(np.mean(lml_rbf))

        # Report a same-data LML ratio where both entries exist.
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
        for scale, ikey, rkey in common_scale_pairs:
            if ikey in rec["lml"] and rkey in rec["lml"]:
                rec["lml_ratio"][f"{scale:g}"] = rec["lml"][ikey] - rec["lml"][rkey]

        rows.append(rec)

        print(
            f"delta={delta:5.2f}  "
            f"regret inv={rec['regret']['invariant']:.4f}"
            f"+/-{rec['regret_se']['invariant']:.4f}  "
            f"rbf={rec['regret']['rbf']:.4f}"
            f"+/-{rec['regret_se']['rbf']:.4f}  "
            f"orbit-gap inv={rec['orbit_gap']['invariant']:.4f} "
            f"rbf={rec['orbit_gap']['rbf']:.4f}  "
            f"true-reward-gap={true_reward_gap:.4f}"
        )

    # Behaviour-change diagnostic.
    identical = total = 0
    for seq in trajectories.values():
        base = dict(seq).get(float(args.deltas[0]))
        for d, traj in seq:
            if d == float(args.deltas[0]):
                continue
            total += 1
            identical += int(traj == base)

    print(
        f"\ntrajectory identical to delta={args.deltas[0]} "
        f"in {identical}/{total} kernel-seed comparisons"
    )

    print("\n[4/4] Writing results and plots...", flush=True)

    with open(args.out, "w") as fh:
        json.dump(
            {
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
            },
            fh,
            indent=2,
        )

    print(f"written {args.out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = [x["delta"] for x in rows]

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        for kind, marker, label in (
            ("invariant", "o-", "repo invariant kernel"),
            ("rbf", "s-", "repo RBF"),
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

        for kind, marker in (
            ("invariant", "o-"),
            ("rbf", "s-"),
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
            label="reward gap",
        )
        ax[1].set_xlabel("reward violation δ")
        ax[1].set_ylabel("predicted Q gap across 180° orbit")
        ax[1].set_title("can the model represent the distinction?")
        ax[1].legend()

        # LML ratio at the invariant scale and RBF scale where available.
        for scale in (args.inv_len_scale, args.rbf_len_scale):
            key = f"{scale:g}"
            vals = [x["lml_ratio"].get(key, np.nan) for x in rows]
            ax[2].plot(d, vals, "o-", label=f"scale {key}")

        ax[2].axhline(0.0, color="k", lw=0.8)
        ax[2].set_xlabel("reward violation δ")
        ax[2].set_ylabel("LML(invariant) − LML(RBF)")
        ax[2].set_title("relative model evidence")
        ax[2].legend()

        fig.tight_layout()
        png = args.out.replace(".json", ".png")
        fig.savefig(png, dpi=150)
        print(f"written {png}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
