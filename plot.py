import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from experiment_grid import ExperimentGrid
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[1]
    mean = np.mean(a, axis=0)
    if a.shape[0] > 1:
        se = scipy.stats.sem(a, axis=0)
        ci = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return mean, ci
    else:
        return mean, [0] * mean.shape[0]

config = {
    "env_id": [
        "SafeCartPole-ArushiModify-v0",
    ],
    "algo": {
        "SPMA-PD": {
            "eta": [0.3, 0.5, 0.7],
            "n_steps": [2048],
            "batch_size": [2048],
            "total_timesteps": [2048 * 50],
            "num_inner_updates": [1],
            "use_armijo_critic": [True],
            "use_armijo_actor": [True],
            "n_epochs": [10],
            "lambda_lr": [0.1, 0.3, 0.5],
            "cost_limit": [25],
        },
        # "SPMA-ALM": {
        #     "eta": [0.3, 0.5, 0.7],
        #     "n_steps": [2048],
        #     "batch_size": [2048],
        #     "total_timesteps": [2048 * 50],
        #     "num_inner_updates": [5],
        #     "use_armijo_critic": [True],
        #     "use_armijo_actor": [True],
        #     "n_epochs": [10],
        #     "cost_limit": [25],
        #     "tau": [0.1, 5, 10],
        # },
    },
    "seed": [10, 20, 30],
}

experiment_name = "spma_test_cost_limit_25"


config = {
    "env_id": [
        "SafeCartPole-ArushiModify-v0",
    ],
    "algo": {
        # "SPMA-PD": {
        #     "eta": [0.3, 0.5, 0.7],
        #     "n_steps": [2048],
        #     "batch_size": [2048],
        #     "total_timesteps": [2048 * 50],
        #     "num_inner_updates": [1],
        #     "use_armijo_critic": [True],
        #     "use_armijo_actor": [True],
        #     "n_epochs": [10],
        #     "lambda_lr": [0.1, 0.3, 0.5],
        #     "cost_limit": [25],
        # },
        "SPMA-ALM": {
            "eta": [0.1],
            "n_steps": [2048],
            "batch_size": [2048],
            "total_timesteps": [2048 * 50],
            "num_inner_updates": [5],
            "use_armijo_critic": [True],
            "use_armijo_actor": [True],
            "n_epochs": [10],
            "cost_limit": [25],
            "tau": [0.001, 0.01, 0.1, 5, 10],
        },
    },
    "seed": [10, 20, 30],
}

experiment_name = "spma_test_cost_limit_25_3"



eg = ExperimentGrid(experiment_name, config)
env_id = eg.config["env_id"][0]

for algo in eg.config['algo']:
    print(eg.variants_for_env_id_algo(env_id, algo))
    paths = [eg.hash_varient(var) for var in eg.variants_for_env_id_algo(env_id, algo)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for var in eg.variants_for_env_id_algo(env_id, algo):
        hash = eg.hash_varient(var)
        path = os.path.join(eg.log_dir, hash)
        eta = var["eta"]
        if algo == 'SPMA-PD':
            lambda_lr = var["lambda_lr"]
            label=f"{eta=}-{lambda_lr=}"
        else:
            tau = var["tau"]
            label=f"{eta=}-{tau=}"

        # plot against timesteps
        eval_files = glob.glob(os.path.join(path, "*npz"))
        logs = [np.load(eval_file) for eval_file in eval_files]
        timesteps = logs[0]["timesteps"]
        returns = np.array([log["results"] for log in logs]).mean(axis=2)
        costs = np.array([log["costs"] for log in logs]).mean(axis=2)
        lambdas = np.array([log["lambda"] for log in logs])

        mean_returns, mean_returns_ci = mean_confidence_interval(returns)
        axs[0].plot(timesteps, mean_returns, label=label)
        axs[0].fill_between(timesteps, mean_returns - mean_returns_ci, mean_returns + mean_returns_ci, alpha=0.4, linewidth=0)
        axs[0].set_xlabel("Timesteps")
        axs[0].set_ylabel("Returns")
        # axs[0].legend()

        mean_costs, mean_costs_ci = mean_confidence_interval(costs - var['cost_limit'])
        axs[1].plot(timesteps, mean_costs, label=label)
        axs[1].fill_between(timesteps, mean_costs - mean_costs_ci, mean_costs + mean_costs_ci, alpha=0.4, linewidth=0)
        axs[1].set_xlabel("Timesteps")
        axs[1].set_ylabel("Constraint Violation")
        axs[1].hlines(y=0, xmin=0, xmax=timesteps[-1], label="b", color="black", linestyles="--")

        mean_lambdas, mean_lambdas_ci = mean_confidence_interval(lambdas)

        axs[2].plot(timesteps, mean_lambdas, label=label)
        axs[2].fill_between(timesteps, mean_lambdas - mean_lambdas_ci, mean_lambdas + mean_lambdas_ci, alpha=0.4, linewidth=0)
        axs[2].set_xlabel("Timesteps")
        axs[2].set_ylabel("Lambda")

    # Move legends to bottom of each subplot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
            fontsize='large', frameon=True, fancybox=True, shadow=True, 
            ncol=4, columnspacing=2, handlelength=3, handletextpad=1)

    fig.suptitle(f"{algo} - Safe Cartpole", fontsize=16, fontweight='bold', y=0.98)
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.9)

    plt.savefig(f"{experiment_name}-{algo}", bbox_inches='tight', pad_inches=0.3)