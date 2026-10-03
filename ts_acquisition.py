#!/usr/bin/env python3
"""
Acquisition-rule variants for the KRVI approximate-invariance experiments,
plus the orbit-coverage diagnostic they exist to measure.

Motivation
----------
The occupancy result (regret at delta=1 is 27x higher when the violated state
is visited 38% of the time than when it is visited 0.4% of the time, while the
LML ratio and the orbit gap both report nothing) was produced under the repo's
optimistic value iteration. Optimism is known to be brittle to misspecification
in exactly this way, so the result is open to the objection that it is a
property of UCB rather than of the invariant prior. Thompson sampling
randomises instead of inflating, and is often argued to be more robust for that
reason.

This module supplies the acquisition rules needed to settle that, holding the
kernel, the environment, the Bellman recursion and the episode bookkeeping
fixed:

    ucb     mean + beta * sd, in both the Bellman target and action selection.
            Identical in behaviour to repo.KRVI; kept here so all three arms
            run through the same code path.

    greedy  Posterior mean in both places; beta is ignored. This is the control
            that separates "randomisation changed the outcome" from "dropping
            the optimism bonus out of the Bellman target changed the outcome".
            Without it, a TS-vs-UCB difference is confounded.

    ts      Posterior mean in the Bellman target; action selection is greedy
            with respect to ONE JOINT posterior sample per (episode, step),
            drawn over the entire finite state-action set. The draw is joint
            and is reused for the whole step, so this is posterior sampling in
            the sense of Bayrooti et al. rather than per-decision noise
            injection.

Why train() is overridden rather than just predict_with_gp()
------------------------------------------------------------
repo.KRVI calls predict_with_gp(...)[0] at two sites -- the Bellman bootstrap
target and action selection -- and the TS arm needs different values at each.
The loop below is a transcription of repo.KRVI.train with those two sites
routed through _target_values() and _behaviour_values(); everything else,
including the order of operations and the W&B logging, is unchanged. Diff it
against the repo method when the repo moves.
"""

from __future__ import annotations

import numpy as np
import torch
import gpytorch
import wandb

import KRVI_algo_test_rotated_invariant_fulloptim as repo

preprocess_state = repo.preprocess_state

ACQUISITIONS = ("ucb", "greedy", "ts")


class AcquisitionKRVI(repo.KRVI):
    """repo.KRVI with a selectable acquisition rule. See module docstring."""

    def __init__(self, *args, acquisition: str = "ucb", coords=None, **kwargs):
        if acquisition not in ACQUISITIONS:
            raise ValueError(
                f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}"
            )

        self.acquisition = acquisition

        # Number of times a joint draw failed and fell back to marginal
        # sampling. Surfaced so a silent change of algorithm is impossible.
        self.ts_fallbacks = 0

        self._episode_draw: dict[int, np.ndarray] = {}

        super().__init__(*args, **kwargs)

        self._coords = None
        self._sa_grid_t = None

        if acquisition == "ts":
            if coords is None:
                raise ValueError(
                    "acquisition='ts' needs coords=<state coordinate array>, "
                    "because the joint draw is taken over the whole finite "
                    "state-action set."
                )

            self._coords = np.asarray(coords, dtype=float)
            self._n_states = len(self._coords)
            self._n_actions = int(self.env.action_space.n)

            # State-major ordering: row index == s * n_actions + a, so a draw
            # reshapes to (n_states, n_actions).
            grid = np.array(
                [
                    np.concatenate(
                        [self._coords[s], self.action_transformation(a)]
                    )
                    for s in range(self._n_states)
                    for a in range(self._n_actions)
                ],
                dtype=float,
            )

            self._sa_grid_t = torch.tensor(
                grid, dtype=torch.float32, device=repo.device
            )

    # -- the two decision points -------------------------------------------

    def _target_values(self, model, states_batch, actions_batch):
        """Values used for the Bellman bootstrap target."""
        acquisition_values, mean, _ = self.predict_with_gp(
            model, states_batch, actions_batch
        )

        return acquisition_values if self.acquisition == "ucb" else mean

    def _behaviour_values(self, model, episode, h, state, action_space, actions_batch):
        """Values the agent acts on at step h."""
        if self.acquisition in ("ucb", "greedy"):
            states_batch = np.tile(state, (len(action_space), 1))

            acquisition_values, mean, _ = self.predict_with_gp(
                model, states_batch, actions_batch
            )

            return acquisition_values if self.acquisition == "ucb" else mean

        draw = self._episode_draw.get(h)

        if draw is None:
            draw = self._joint_posterior_draw(model, episode, h)
            self._episode_draw[h] = draw

        return draw[self._state_index(state)]

    # -- Thompson sampling machinery ---------------------------------------

    def _state_index(self, state) -> int:
        state = np.asarray(state, dtype=float).reshape(1, -1)

        return int(np.argmin(((self._coords - state) ** 2).sum(axis=1)))

    def _draw_seed(self, episode: int, h: int) -> int:
        """
        Deterministic key per (run seed, episode, step).

        Seeding immediately before each draw, rather than once per run, makes
        the TS stream independent of anything else that touches the global
        torch RNG -- so a run is reproducible even if the GP code starts
        consuming randomness.
        """
        base = 0 if self.seed is None else int(self.seed)

        return int((base * 1_000_003 + episode * 1_009 + h) % (2**31 - 1))

    def _joint_posterior_draw(self, model, episode: int, h: int) -> np.ndarray:
        """
        One joint sample of Q_h over the whole finite state-action set.

        Joint, not marginal: the sample is consistent across states and
        actions, which is what makes acting greedily with respect to it
        posterior sampling.

        model.posterior() is used rather than model(...) on purpose. botorch
        applies a Standardize outcome transform to SingleTaskGP by default
        (0.12 onwards -- this repo's requirements.txt pins 0.10.0 but the runs
        were made on 0.18.1), so model(...) returns standardised values while
        the Bellman targets live in the original scale. Only posterior() maps
        back.
        """
        model.eval()
        model.likelihood.eval()

        torch.manual_seed(self._draw_seed(episode, h))

        try:
            with torch.no_grad(), gpytorch.settings.fast_computations(
                covar_root_decomposition=False
            ):
                posterior = model.posterior(self._sa_grid_t)
                sample = posterior.rsample(torch.Size([1]))

            values = sample.reshape(-1).detach().cpu().numpy()

        except Exception:
            # Independent marginal draws: NOT posterior sampling. Counted so
            # the driver can refuse to report a run that fell back.
            self.ts_fallbacks += 1

            with torch.no_grad():
                posterior = model.posterior(self._sa_grid_t)
                mean = posterior.mean.reshape(-1).detach().cpu().numpy()
                sd = posterior.variance.sqrt().reshape(-1).detach().cpu().numpy()

            rng = np.random.default_rng(self._draw_seed(episode, h))
            values = mean + sd * rng.standard_normal(mean.shape)

        return values.reshape(self._n_states, self._n_actions)

    # -- training loop ------------------------------------------------------

    def train(self, T: int):
        action_space = np.arange(self.env.action_space.n)

        actions_batch = np.array(
            [self.action_transformation(action) for action in action_space]
        )

        if self.logging:
            wandb.run.summary["episode length"] = self.horizon
            wandb.run.summary["iterations"] = T
            wandb.run.summary["acquisition"] = self.acquisition

        all_states = []
        all_actions = []
        all_rewards = []
        Qt = [None] * self.horizon
        cumulative_returns = []

        for episode in range(T):
            if self.verbose > 0:
                print(f"Episode {episode}")

            if episode > 0:
                for h in reversed(range(len(all_states[-1]))):
                    X_states = []
                    X_actions = []
                    y_values = []

                    for i in range(episode):
                        if h < len(all_states[i]):
                            X_states.append(all_states[i][h])
                            X_actions.append(all_actions[i][h])

                            if h < len(all_states[i]) - 1:
                                next_state = all_states[i][h + 1]

                                states_expanded = np.tile(
                                    next_state, (len(action_space), 1)
                                )

                                max_q_value = 0

                                if Qt[h + 1] is not None:
                                    max_q_value = np.max(
                                        self._target_values(
                                            Qt[h + 1],
                                            states_expanded,
                                            actions_batch,
                                        )
                                    )

                                Qnext = max_q_value
                            else:
                                Qnext = 0

                            y_values.append(all_rewards[i][h] + Qnext)

                    if X_states:
                        X = np.column_stack((X_states, X_actions))
                        y = np.array(y_values)
                        Qt[h] = self.GP_regression_torch(X, y)

            # One draw per step, reused for the whole episode.
            self._episode_draw = {}

            episode_states = []
            episode_actions = []
            episode_rewards = []

            initial_state, info = self.env.reset()

            state = preprocess_state(initial_state)

            for h in range(self.horizon):
                if Qt[h] is not None:
                    q_values = self._behaviour_values(
                        Qt[h], episode, h, state, action_space, actions_batch
                    )
                else:
                    q_values = np.zeros(len(action_space))

                action = action_space[np.argmax(q_values)]

                next_state, reward, done, truncated, info = self.env.step(action)

                next_state = preprocess_state(next_state)

                episode_states.append(state)
                action = self.action_transformation(action)
                episode_actions.append(action)
                episode_rewards.append(reward)

                if done or truncated:
                    break

                state = next_state

            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))

            episode_cum_rewards = np.sum(episode_rewards)
            cumulative_returns.append(episode_cum_rewards)

            if self.logging:
                wandb.log(
                    {
                        "Episode_number": episode,
                        "Episode_Rewards": episode_cum_rewards,
                        "cumulative_returns": sum(cumulative_returns),
                    }
                )


# ---------------------------------------------------------------------------
# Orbit residual contrast
# ---------------------------------------------------------------------------


def orbit_residual_contrast(
    gp_calls,
    coords,
    cell,
    orbit_pairs,
    action_transformation,
):
    """
    Signed residual contrast between the violated state-action and the rest of
    its orbit, on the Bellman regression data the run actually collected.

    Why this and not orbit_gap: an invariant kernel pins the posterior equal
    across an orbit by construction, so orbit_gap is ~1e-7 whatever the agent
    does and cannot report a violation. But a violation at one orbit member
    must still leave a SIGNED footprint in the residuals -- the pooled fit sits
    between the violated target and its partners, under-predicting at one and
    over-predicting at the others. That footprint is what an external audit can
    see, and it only exists if the agent collected data at both.

    Returns a dict with the per-side counts and means, the contrast, and a
    standardised contrast

        z = (mean_violated - mean_orbit) / sqrt(var_v / n_v + var_o / n_o)

    `z` is None when either side has fewer than two observations. Read it as a
    standardised effect size, not a p-value: rewards in this environment are
    deterministic, so the residual spread is model misfit rather than
    observation noise, and the usual sampling-theory interpretation does not
    apply.

    `n_violated == 0` is a result, not a missing value: it means the
    acquisition rule never went there.
    """
    coords = np.asarray(coords, dtype=float)

    vs, va = int(cell[0]), int(cell[1])

    # Matched at STATE level (first two columns of X), pooling over whichever
    # action was taken there. Matching the exact violated (state, action) pair
    # instead would leave the statistic undefined almost always: the violated
    # action is the runner-up, so a greedy or optimistic learner essentially
    # never collects data at that pair, and n < 2 gives no contrast. The
    # state-level contrast is still a valid test of pooling bias -- under exact
    # invariance the residual distribution must be identical across the states
    # of an orbit -- and it has enough data to be computed. Pair-level coverage
    # is reported separately as violated_pair_visits by decompose_regret.
    state_violated = np.asarray(coords[vs], dtype=float)

    states_orbit = [
        np.asarray(coords[int(s)], dtype=float)
        for s in sorted({int(s) for (s, _) in orbit_pairs if int(s) != vs})
    ]

    resid_violated: list[float] = []
    resid_orbit: list[float] = []

    for call in gp_calls:
        X = np.asarray(call["X"], dtype=float)
        y = np.asarray(call["y"], dtype=float).reshape(-1)

        if X.size == 0:
            continue

        model = call["gp"]
        model.eval()
        model.likelihood.eval()

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=repo.device)
            prediction = (
                model.posterior(X_t).mean.reshape(-1).detach().cpu().numpy()
            )

        residual = y - prediction

        states = X[:, :2]

        hit_violated = np.all(np.isclose(states, state_violated, atol=1e-8), axis=1)

        resid_violated.extend(residual[hit_violated].tolist())

        for state in states_orbit:
            hit = np.all(np.isclose(states, state, atol=1e-8), axis=1)
            resid_orbit.extend(residual[hit].tolist())

    def mean_of(values):
        return float(np.mean(values)) if values else None

    n_v, n_o = len(resid_violated), len(resid_orbit)

    mean_v, mean_o = mean_of(resid_violated), mean_of(resid_orbit)

    contrast = None if (mean_v is None or mean_o is None) else mean_v - mean_o

    z = None

    if n_v > 1 and n_o > 1:
        se = np.sqrt(
            np.var(resid_violated, ddof=1) / n_v
            + np.var(resid_orbit, ddof=1) / n_o
        )

        z = float(contrast / se) if se > 0 else None

    return {
        "n_rows_violated_state": n_v,
        "n_rows_orbit_states": n_o,
        "mean_residual_violated_state": mean_v,
        "mean_residual_orbit_states": mean_o,
        "contrast": contrast,
        "z": z,
    }
