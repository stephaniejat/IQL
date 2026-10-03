# Retrospective uploads to W&B of four run experiments

import json
import wandb


ENTITY = "stephanie-jat-ucl"
PROJECT = "kvri-trials"


def upload_experiment(filename, mode, name, tags):
    # Load the existing experiment results
    with open(filename, "r") as f:
        data = json.load(f)

    # Create one W&B run representing this completed experiment
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        tags=tags,
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
        run.log(
            {
                "delta": row["delta"],
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

    # Attach the original JSON to the W&B run
    run.save(filename)

    run.finish()


# Exact-symmetry baseline
upload_experiment(
    "baseline.json",
    mode="oracle",
    name="oracle-baseline",
    tags=[
        "retrospective",
        "KRVI-baseline",
        "oracle",
    ],
)

# Point-violation experiment
upload_experiment(
    "point_violation.json",
    mode="point",
    name="point-violation",
    tags=[
        "retrospective",
        "point-violation",
    ],
)

# # Relevant violation experiment
# upload_experiment(
#     "relevant.json",
#     mode="relevant",
#     name="decision-relevant-violation",
#     tags=[
#         "retrospective",
#         "relevant-violation",
#     ],
# )

# # Irrelevant violation experiment
# upload_experiment(
#     "irrelevant.json",
#     mode="irrelevant",
#     name="decision-irrelevant-violation",
#     tags=[
#         "retrospective",
#         "irrelevant-violation",
#     ],
# )
