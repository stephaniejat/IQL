# Retrospective uploads to wandb

import json
import wandb


ENTITY = "stephanie-jat-ucl"
PROJECT = "kvri-trials"


def upload_experiment(filename):
    # Load the existing experiment results
    with open(filename, "r") as f:
        data = json.load(f)

    mode = data["mode"]

    # Create one W&B run representing this completed experiment
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"decision-{mode}-violation",
        tags=[
            "retrospective",
            "KRVI-baseline",
            mode,
        ],
        config={
            "algorithm": data["algorithm"],
            "environment": data["environment"],
            "violation_mode": mode,
            "H": data["H"],
            "T": data["T"],
            "beta": data["beta"],
            "noise": data["noise"],
            "invariant_length_scale": data["length_scales"]["invariant"],
            "rbf_length_scale": data["length_scales"]["rbf"],
            "baseline_symmetry_error": data["baseline_symmetry_error"],
            "source": filename,
        },
    )

    # Log each delta condition as a W&B step
    for row in data["rows"]:
        delta = row["delta"]

        run.log(
            {
                "delta": delta,
                # Regret
                "regret/invariant": row["regret"]["invariant"],
                "regret/rbf": row["regret"]["rbf"],
                "regret_se/invariant": row["regret_se"]["invariant"],
                "regret_se/rbf": row["regret_se"]["rbf"],
                # Structural diagnostic
                "orbit_gap/invariant": row["orbit_gap"]["invariant"],
                "orbit_gap/rbf": row["orbit_gap"]["rbf"],
                # True reward distinction
                "true_reward_gap": row["true_reward_gap"],
                # Decision consequence
                "optimal_action_before": row["optimal_action_before"],
                "optimal_action_after": row["optimal_action_after"],
                "decision_changed": int(row["decision_changed"]),
                # Model evidence
                "lml/invariant_0.5": row["lml"]["invariant@0.5"],
                "lml/rbf_0.5": row["lml"]["rbf@0.5"],
                "lml/invariant_0.1": row["lml"]["invariant@0.1"],
                "lml/rbf_0.1": row["lml"]["rbf@0.1"],
                # LML differences
                "lml_ratio/0.5": row["lml_ratio"]["0.5"],
                "lml_ratio/0.1": row["lml_ratio"]["0.1"],
            }
        )

    # Preserve the original JSON as a file attached to the run
    run.save(filename)

    run.finish()


upload_experiment("relevant.json")
upload_experiment("irrelevant.json")
