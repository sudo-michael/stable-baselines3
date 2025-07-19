import glob
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from experiment_grid import ExperimentGrid
#  https://arxiv.org/pdf/1603.05738#page=2.51

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
            "n_epochs": [1],
            "lambda_lr": [0.1, 0.3, 0.5],
            "cost_limit": [25],
        },
        # "SPMA": {
        #     "eta": [0.3, 0.5, 0.7, 0.9],
        #     "n_steps": [2048],
        #     "batch_size": [2048],
        #     "total_timesteps": [2048 * 4 * 25],
        #     "use_armijo_critic": [True],
        #     "use_armijo_actor": [True],
        #     "n_epochs": [5],
        # },
    },
    "seed": [0, 1, 2],
}
experiment_name = "spma_test_cost_limit_25_n_epochs_1_2"
eg = ExperimentGrid(experiment_name, config)
env_id = eg.config['env_id'][0]

print(eg.variants_for_env_id_algo(env_id, "SPMA-PD"))
paths = [eg.hash_varient(var) for var in eg.variants_for_env_id_algo(env_id, "SPMA-PD")]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for var in eg.variants_for_env_id_algo(env_id, "SPMA-PD"):
    hash = eg.hash_varient(var)
    path = os.path.join(eg.log_dir, hash)
    eta = var['eta']
    lambda_lr = var['lambda_lr']
    # plot against timesteps
    eval_files = glob.glob(os.path.join(path, "*npz"))
    logs = np.load(eval_files[0])
    timesteps = logs["timesteps"]
    returns = np.array(logs["results"]).mean(axis=1)
    costs = np.array(logs["costs"]).mean(axis=1)
    lambdas = np.array(logs["lambda"])

    axs[0].plot(timesteps, returns, label=f"{eta=}-{lambda_lr=}")
    axs[0].set_xlabel("Timesteps")
    axs[0].set_ylabel("Returns")
    axs[0].legend()

    axs[1].plot(timesteps, costs, label=f"{eta=}-{lambda_lr=}")
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Cost Returns")
    axs[1].hlines(y=25, xmin=0, xmax=timesteps[-1], label="b", color="black", linestyles="--")

    axs[2].plot(timesteps, lambdas, label=f"{eta=}-{lambda_lr=}")
    axs[2].set_xlabel("Timesteps")
    axs[2].set_ylabel("Lambda")

plt.savefig(f"{experiment_name}-{env_id}-SPMAPD")
