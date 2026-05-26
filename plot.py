import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_eval_curve(root: Path, variant: str, n_seeds: int):
    """
    Returns:
        timesteps: shape (n_evals,)
        seed_means: shape (n_seeds, n_evals)

    Each seed's evaluation result usually has shape:
        results: (n_evals, n_eval_episodes)

    We first average over eval episodes, giving one curve per seed.
    Then later we average/std over seeds.
    """
    all_seed_means = []
    timesteps_ref = None

    for seed in range(n_seeds):
        eval_path = root / f"task_id_{seed}" / variant / "evaluations.npz"

        if not eval_path.exists():
            raise FileNotFoundError(f"Missing file: {eval_path}")

        print(f"loading from {eval_path}")
        data = np.load(eval_path)
        timesteps = data["timesteps"]
        results = data["results"]

        # Mean reward over eval episodes for each eval point
        seed_mean_reward = results.mean(axis=1)

        if timesteps_ref is None:
            timesteps_ref = timesteps
        else:
            if not np.array_equal(timesteps_ref, timesteps):
                raise ValueError(
                    f"Timestep mismatch for seed {seed}, variant {variant}.\nExpected {timesteps_ref}\nGot      {timesteps}"
                )

        all_seed_means.append(seed_mean_reward)

    seed_means = np.stack(all_seed_means, axis=0)
    return timesteps_ref, seed_means

experiment_id = "atari_pong_eval_object"
root = Path(f"./runs/{experiment_id}")
n_seeds = 3
variants = ["standard", "tertiary"]

summary = {}

for variant in variants:
    timesteps, seed_means = load_eval_curve(root, variant, n_seeds)

    mean_reward = seed_means.mean(axis=0)
    std_reward = seed_means.std(axis=0)

    summary[variant] = {
        "timesteps": timesteps,
        "seed_means": seed_means,
        "mean": mean_reward,
        "std": std_reward,
    }

    print(f"\n{variant}")
    print("timesteps:", timesteps)
    print("mean reward:", mean_reward)
    print("std reward:", std_reward)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, variant in zip(axes, variants):
    timesteps = summary[variant]["timesteps"]
    mean_reward = summary[variant]["mean"]
    std_reward = summary[variant]["std"]

    ax.plot(timesteps, mean_reward, label="PPO")
    ax.fill_between(
        timesteps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        alpha=0.25,
        # label="±1 std over seeds",
    )


    # if variant == "standard":
    #     ax.set_title('Atari Score')
    # ax.set_title(variant)
    ax.set_xlabel("Timesteps")
    # ax.grid(True, alpha=0.3)
    if variant == 'tertiary':
        ax.set_ylim(bottom=-1000, top=0.0)

    ax.legend()

axes[0].set_ylabel("Average Atari Score")
axes[1].set_ylabel("Average Tertiary Return")
plt.suptitle('Pong')

print(f"Plotting to ./plots/{experiment_id}_eval_mean_std_side_by_side.png")
plt.tight_layout()
plt.savefig(f"./plots/{experiment_id}_eval_mean_std_side_by_side.png", dpi=200)
