#!/usr/bin/env python3
"""Approximate invariance in the IQL synthetic setting: benefit, cost, detection.

Drop in the root of mtkresearch/IQL. Reuses that repo's reward and transition
generators, group action, invariant kernel, value iteration and transition
dynamics. The only manipulation is a controlled violation of the assumed
symmetry, swept in magnitude.

WHAT IS MEASURED

regret      tail episodic regret against value iteration on the violated MDP,
            with a standard error over seeds.

orbit gap   |mean(i,j) - mean(g(i,j))| at the violated cell, where g is the
            repo's group action (i,j) -> (n-1-i, m-1-j). This is what the model
            believes the difference across the orbit is. The invariant kernel
            forces it to zero by construction, so comparing it against the true
            gap says whether the violated distinction is representable at all.
            This replaces an earlier |mean - reward| column, which was wrong:
            the regression target is a Bellman value, not a reward, so that
            difference measured the horizon rather than model error.

LMR         log marginal likelihood ratio, invariant minus RBF, fitted to the
            SAME Bellman regression data, computed separately at each tuned
            length scale. Comparing the invariant kernel at one scale against
            RBF at another would confound invariance with smoothness. If LMR
            turns negative at a smaller delta than the regret curves cross, a
            learner could tell from passive data that pooling has become
            unsafe; if not, detection needs active structure testing.

trajectory  whether the visited state-action sequence changes with delta. If it
            does not, the violation is altering value estimates without
            altering behaviour, so regret differences are not being driven by
            exploration.

VIOLATION MODES

point   adds delta at the argmax cell only, so the violated distinction sits
        where the optimal policy goes. Decision-relevant.
spread  adds delta * xi across one representative of every orbit, xi drawn once
        and held fixed across delta. Diffuse; mostly not decision-relevant.

The contrast between the two at matched delta is the empirical content of
"task-relevant equivalence" versus "approximate symmetry": the global defect is
comparable, the decision-localised defect is not.

Usage:
    python3 approximate_invariance.py --quick
    python3 approximate_invariance.py --T 200 --seeds 8 --out point.json
    python3 approximate_invariance.py --mode spread --T 200 --seeds 8 --out spread.json
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import contextlib, io, os


import numpy as np

# The repo imports wandb at module scope; stub it so nothing is logged.
if "wandb" not in sys.modules:
    stub = types.ModuleType("wandb")
    stub.init = lambda *a, **k: None
    stub.log = lambda *a, **k: None
    stub.Image = lambda *a, **k: None
    stub.run = types.SimpleNamespace(summary={})
    sys.modules["wandb"] = stub

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from framework import (
    GroupInvariantKernel,
    group_SA,
    reward_RKHS,
    transition_dynamics,
    transition_P_RKHS,
    value_iteration_episodic,
)


def make_kernel(name: str, length_scale: float):
    if name == "invariant":
        return GroupInvariantKernel(
            base_kernel="RBF", length_scale=length_scale, group=group_SA
        )
    return RBF(length_scale=length_scale, length_scale_bounds="fixed")


def build_environment(n_states, n_actions, p_kernel, alpha, seed):
    np.random.seed(seed)
    state_space = np.linspace(-1, 1, num=n_states).reshape(-1, 1)
    action_space = np.linspace(-1, 1, num=n_actions).reshape(-1, 1)
    # reward_RKHS and transition_P_RKHS print the full transition tensor
    with contextlib.redirect_stdout(io.StringIO()):
        r = reward_RKHS(p_kernel, state_space, action_space, alpha=alpha)
        P = transition_P_RKHS(state_space, action_space, p_kernel, alpha=alpha)
    return state_space, action_space, r, P


def orbit_partner(i: int, j: int, n: int, m: int) -> tuple[int, int]:
    """The repo's group action on grid indices: (s, a) -> (-s, -a)."""

    return n - 1 - i, m - 1 - j


def violate(r: np.ndarray, delta: float, mode: str, rng):
    """Break invariance by delta, at one cell or across half-orbits."""

    n, m = r.shape
    out = r.copy()
    if mode == "point":
        i, j = np.unravel_index(int(np.argmax(r)), r.shape)
        out[i, j] += delta
        return out, (int(i), int(j))

    xi = rng.standard_normal(r.shape)
    xi /= np.abs(xi).max()
    for i in range(n):
        for j in range(m):
            if (i, j) < orbit_partner(i, j, n, m):  # one member per orbit
                out[i, j] += delta * xi[i, j]
    i, j = np.unravel_index(int(np.argmax(r)), r.shape)
    return out, (int(i), int(j))


def krvi(
    state_space,
    action_space,
    r,
    P,
    H,
    T,
    beta,
    kernel_name,
    length_scale,
    noise,
    rng,
    lml_scales=(),
):
    """KRVI-UCB following the repo's pi_krvi_policy, returning values not logs."""

    S, A = len(state_space), len(action_space)
    grid = np.array([np.hstack((s, a)) for s in state_space for a in action_space])
    kernel = make_kernel(kernel_name, length_scale)
    V_star = np.asarray(
        value_iteration_episodic(state_space, action_space, r, P, H=H)
    ).ravel()

    Q_ucb = np.zeros((H + 1, S, A))
    hist_s, hist_a, hist_r = [], [], []
    regrets = []
    last = {"mean": None, "std": None, "X": None, "y": None}

    for episode in range(T):
        if episode > 0:
            for h in reversed(range(H)):
                X = np.stack([[hist_s[i][h], hist_a[i][h]] for i in range(episode)])
                y = []
                for i in range(episode):
                    if h < H - 1:
                        nxt = int(np.argmin(np.abs(state_space - hist_s[i][h + 1])))
                        y.append(hist_r[i][h] + float(np.max(Q_ucb[h + 1][nxt, :])))
                    else:
                        y.append(hist_r[i][h])
                y = np.asarray(y)
                gpr = GaussianProcessRegressor(
                    kernel=kernel, optimizer=None, alpha=noise
                ).fit(X, y)
                mean, std = gpr.predict(grid, return_std=True)
                Q_ucb[h] = mean.reshape(S, A) + beta * std.reshape(S, A)
                if h == 0:
                    last = {
                        "mean": mean.reshape(S, A),
                        "std": std.reshape(S, A),
                        "X": X,
                        "y": y,
                    }

        s_idx = int(rng.integers(S))
        state = np.asarray(state_space[s_idx]).ravel()[:1]
        ep_s, ep_a, ep_r = [], [], []
        for h in range(H):
            i = int(np.argmin(np.abs(state_space - state)))
            a_idx = int(np.argmax(Q_ucb[h][i]))
            action = action_space[a_idx]
            ep_s.append(float(np.asarray(state).ravel()[0]))
            ep_a.append(float(np.asarray(action).ravel()[0]))
            ep_r.append(float(r[i, a_idx]))
            state = np.asarray(
                transition_dynamics(state, action, P, state_space, action_space)
            ).ravel()[:1]
        hist_s.append(np.array(ep_s))
        hist_a.append(np.array(ep_a))
        hist_r.append(np.array(ep_r))
        regrets.append(float(V_star[s_idx] - np.sum(ep_r)))

    result = {"regret": np.asarray(regrets), **last}

    # Fingerprint of the visited state-action sequence, to test whether the
    # violation changed behaviour or only the value estimate.
    result["trajectory"] = tuple(
        (round(float(s), 6), round(float(act), 6))
        for ep_states, ep_actions in zip(hist_s, hist_a)
        for s, act in zip(ep_states, ep_actions)
    )

    # Detection: both kernels on the same data, at each requested length scale.
    if lml_scales and last["X"] is not None:
        result["lml"] = {}
        for ls in lml_scales:
            for name in ("invariant", "rbf"):
                g = GaussianProcessRegressor(
                    kernel=make_kernel(name, ls), optimizer=None, alpha=noise
                )
                g.fit(last["X"], last["y"])
                result["lml"][f"{name}@{ls:g}"] = float(
                    g.log_marginal_likelihood_value_
                )
    return result


def tune(state_space, action_space, r, P, args, kernel_name, rng_seed):
    """Pick each kernel's length scale on its own terms, at delta = 0."""

    best, best_ls = np.inf, args.grid[0]
    for ls in args.grid:
        out = krvi(
            state_space,
            action_space,
            r,
            P,
            args.H,
            args.tune_T,
            args.beta,
            kernel_name,
            ls,
            args.noise,
            np.random.default_rng(rng_seed),
        )
        score = float(out["regret"][-max(1, args.tune_T // 3) :].mean())
        if score < best:
            best, best_ls = score, ls
    return best_ls, best


def stderr(values) -> float:
    return (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else float("nan")
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=int, default=10)
    p.add_argument("--actions", type=int, default=10)
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--tune_T", type=int, default=40)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=1e-10)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--p_kernel", default="RBF")
    p.add_argument("--mode", choices=["point", "spread"], default="point")
    p.add_argument(
        "--deltas", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.2, 0.4]
    )
    p.add_argument("--grid", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="approximate_invariance_results.json")
    a = p.parse_args()
    if a.quick:
        a.T, a.tune_T, a.seeds = 25, 12, 1
        a.deltas, a.grid = [0.0, 0.2], [0.5, 1.0]

    state_space, action_space, r0, P = build_environment(
        a.states, a.actions, a.p_kernel, a.alpha, 0
    )

    ls = {}
    for name in ("invariant", "rbf"):
        ls[name], score = tune(state_space, action_space, r0, P, a, name, 0)
        print(
            f"tuned {name:10s} length_scale={ls[name]}  "
            f"(delta=0 tail regret {score:.4f})"
        )
    lml_scales = tuple(sorted({ls["invariant"], ls["rbf"]}))
    print(f"detection compares both kernels at length scales {lml_scales}\n")

    rows = []
    trajectories: dict[tuple[str, int], list] = {}
    for delta in a.deltas:
        rec = {
            "delta": delta,
            "regret": {},
            "regret_se": {},
            "sd": {},
            "orbit_gap": {},
            "true_gap": None,
            "lml": {},
        }
        for name in ("invariant", "rbf"):
            reg, sd, gap, lm = [], [], [], []
            for seed in range(a.seeds):
                ss, asp, r, PP = build_environment(
                    a.states, a.actions, a.p_kernel, a.alpha, seed
                )
                r_v, cell = violate(r, delta, a.mode, np.random.default_rng(0))
                out = krvi(
                    ss,
                    asp,
                    r_v,
                    PP,
                    a.H,
                    a.T,
                    a.beta,
                    name,
                    ls[name],
                    a.noise,
                    np.random.default_rng(seed),
                    lml_scales=lml_scales,
                )
                reg.append(float(out["regret"][-max(1, a.T // 5) :].mean()))
                trajectories.setdefault((name, seed), []).append(
                    (delta, out["trajectory"])
                )
                if out["mean"] is not None:
                    i, j = cell
                    gi, gj = orbit_partner(i, j, a.states, a.actions)

                    sd.append(float(out["std"][i, j]))

                    # Difference in the learned Bellman value across the orbit.
                    # The invariant kernel forces this to zero by construction.
                    gap.append(float(abs(out["mean"][i, j] - out["mean"][gi, gj])))
                if "lml" in out:
                    lm.append(out["lml"])
            rec["regret"][name] = float(np.mean(reg))
            rec["regret_se"][name] = stderr(reg)
            rec["sd"][name] = float(np.mean(sd)) if sd else None
            rec["orbit_gap"][name] = float(np.mean(gap)) if gap else None
            if lm:
                for key in lm[0]:
                    rec["lml"][key] = float(np.mean([d[key] for d in lm]))
        rows.append(rec)

        inv, rbf = rec["regret"]["invariant"], rec["regret"]["rbf"]
        lmr = {
            f"{s:g}": rec["lml"].get(f"invariant@{s:g}", np.nan)
            - rec["lml"].get(f"rbf@{s:g}", np.nan)
            for s in lml_scales
        }
        lmr_txt = "  ".join(f"LMR@{k} {v:+8.2f}" for k, v in lmr.items())
        print(
            f"delta={delta:5.2f}  "
            f"regret inv {inv:6.4f}+/-{rec['regret_se']['invariant']:.4f}  "
            f"rbf {rbf:6.4f}+/-{rec['regret_se']['rbf']:.4f}  "
            f"gap inv {rec['orbit_gap']['invariant']:.4f} "
            f"rbf {rec['orbit_gap']['rbf']:.4f} true {rec['true_gap']:.4f}  "
            f"{lmr_txt}"
        )

    identical = total = 0
    for seq in trajectories.values():
        base = dict(seq).get(a.deltas[0])
        for d, traj in seq:
            if d == a.deltas[0]:
                continue
            total += 1
            identical += int(traj == base)
    print(f"\ntrajectory identical to delta={a.deltas[0]} in {identical}/{total} runs")
    if total and identical == total:
        print(
            "  behaviour never changed: the violation moved value estimates "
            "only, so regret differences are not driven by exploration. "
            "Consider raising --beta."
        )

    cross_r = next(
        (x["delta"] for x in rows if x["regret"]["invariant"] > x["regret"]["rbf"]),
        None,
    )
    print(f"regret crossover at delta ~ {cross_r}")
    for s in lml_scales:
        cross_l = next(
            (
                x["delta"]
                for x in rows
                if (x["lml"].get(f"invariant@{s:g}", 0) - x["lml"].get(f"rbf@{s:g}", 0))
                < 0
            ),
            None,
        )
        print(f"detection crossover at length scale {s:g}: delta ~ {cross_l}")

    with open(a.out, "w") as fh:
        json.dump(
            {
                "length_scales": ls,
                "lml_scales": list(lml_scales),
                "mode": a.mode,
                "rows": rows,
            },
            fh,
            indent=1,
        )
    print(f"written {a.out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = [x["delta"] for x in rows]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for name, marker, label in (
            ("invariant", "o-", "invariant kernel"),
            ("rbf", "s-", "RBF"),
        ):
            ax[0].errorbar(
                d,
                [x["regret"][name] for x in rows],
                yerr=[x["regret_se"][name] for x in rows],
                fmt=marker,
                capsize=3,
                label=label,
            )
        ax[0].set_xlabel("symmetry violation $\\delta$")
        ax[0].set_ylabel("tail episodic regret")
        ax[0].legend()
        ax[0].set_title(f"benefit and cost of pooling ({a.mode})")

        ax[1].plot(d, [x["true_gap"] for x in rows], "k--", label="true")
        for name, marker in (("invariant", "o-"), ("rbf", "s-")):
            ax[1].plot(d, [x["orbit_gap"][name] for x in rows], marker, label=name)
        ax[1].set_xlabel("symmetry violation $\\delta$")
        ax[1].set_ylabel("believed gap across the orbit")
        ax[1].legend()
        ax[1].set_title("is the distinction representable?")

        ax[2].axhline(0, color="k", lw=0.8)
        for s in lml_scales:
            ax[2].plot(
                d,
                [
                    x["lml"].get(f"invariant@{s:g}", np.nan)
                    - x["lml"].get(f"rbf@{s:g}", np.nan)
                    for x in rows
                ],
                "o-",
                label=f"length scale {s:g}",
            )
        ax[2].set_xlabel("symmetry violation $\\delta$")
        ax[2].set_ylabel("log marginal likelihood ratio")
        ax[2].legend()
        ax[2].set_title("can the learner tell?")
        fig.tight_layout()
        fig.savefig(a.out.replace(".json", ".png"), dpi=150)
        print(f"written {a.out.replace('.json', '.png')}")
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
