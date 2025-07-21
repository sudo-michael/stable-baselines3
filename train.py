import os
import safe_simple_gymnasium
import gymnasium as gym
from experiment_grid import ExperimentGrid
from stable_baselines3.common.callbacks import CostEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.spma.spma import SPMA
from stable_baselines3.spma_pd.spma_pd import SPMAPD
from stable_baselines3.spma_alm.spma_alm import SPMAALM


def train(
    experiment_id,
    exp_log_dir,
    algo,
    env_id,
    seed,
    algo_params,
):
    print(f"training: {experiment_id=}")
    env = make_vec_env(env_id, seed=seed)
    eval_env = make_vec_env(env_id, seed=seed + 1337)

    total_timesteps = algo_params.pop("total_timesteps")
    if algo == "SPMA-PD":
        num_inner_updates = algo_params.pop("num_inner_updates")
        callback = CostEvalCallback(
            eval_env,
            eval_freq=algo_params["n_steps"] * num_inner_updates + 1,
            log_path=exp_log_dir,
            deterministic=False,
            verbose=0,
        )
        model = SPMAPD("MlpPolicy", env, verbose=1, seed=seed, device="cuda", **algo_params)
        model.learn(
            total_timesteps=total_timesteps + 1, num_inner_updates=num_inner_updates, progress_bar=True, callback=callback
        )
    elif algo == "SPMA-ALM":
        num_inner_updates = algo_params.pop("num_inner_updates")
        callback = CostEvalCallback(
            eval_env,
            eval_freq=algo_params["n_steps"] * num_inner_updates + 1,
            log_path=exp_log_dir,
            deterministic=False,
            verbose=0,
        )
        model = SPMAALM("MlpPolicy", env, verbose=1, seed=seed, device="cuda", **algo_params)
        model.learn(
            total_timesteps=total_timesteps + 1, num_inner_updates=num_inner_updates, progress_bar=True, callback=callback
        )
    else:
        model = SPMA("MlpPolicy", env, verbose=1, seed=seed, device="cuda", **algo_params)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(os.path.join(exp_log_dir, f"final_model_{seed}"))

    return True


if __name__ in "__main__":
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
                "tau": [0.001, 0.01],
            },
        },
        "seed": [10, 20, 30],
    }

    eg = ExperimentGrid("spma_test_cost_limit_25_4", config)
    eg.run(train, num_pool=4)
