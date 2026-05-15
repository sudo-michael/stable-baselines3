import os
import safe_simple_gymnasium
import gymnasium as gym
import safety_gymnasium
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.spma_pd.spma_pd import SPMAPD
from stable_baselines3.spma_alm.spma_alm import SPMAALM

class RolloutCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rollout_info_buffer = []

    def _on_rollout_start(self) -> None:
        self.rollout_info_buffer = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            if maybe_ep_info is not None:
                self.rollout_info_buffer.extend([maybe_ep_info])
        return True

def make_env(env_id):
    if "CartPole" in env_id:
        env = gym.make(env_id)
    else:
        safe_env = safety_gymnasium.make(env_id)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)
    return env


def train(
    experiment_id,
    exp_log_dir,
    algo,
    env_id,
    seed,
    algo_params,
):
    print(f"training: {experiment_id=}")

    env = make_env(env_id)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env.seed(seed)

    eval_env = make_env(env_id)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env.seed(seed)

    logger = configure(exp_log_dir, ["stdout", "csv", "tensorboard"], f"{seed}")

    total_timesteps = algo_params.pop("total_timesteps")
    num_inner_updates = algo_params.pop("num_inner_updates")
    if algo == "SPMA-PD":
        model = SPMAPD(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            device="cuda",
            **algo_params,
            tensorboard_log=exp_log_dir,
            eval_env=eval_env,
            n_eval_episodes=10,
            deterministic=False,
        )
        callback=None
    elif algo == "SPMA-ALM":
        model = SPMAALM(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            device="cuda",
            **algo_params,
            tensorboard_log=exp_log_dir,
            eval_env=eval_env,
            n_eval_episodes=10,
            deterministic=False,
        )
        callback = RolloutCallback()

    model.set_logger(logger)
    model.learn(total_timesteps=total_timesteps, num_inner_updates=num_inner_updates, callback=callback)
    model.save(os.path.join(exp_log_dir, f"final_model_{seed}"))

    return True


if __name__ in "__main__":
    # env_id = "SafetyHopperVelocity-v1"
    env_id = "SafeCartPole-ArushiModify-v0"
    algo_params = {
        "eta": 0.1,
        "n_steps": 2048,
        "batch_size": 2048,
        "total_timesteps": 2048 * 50,
        "num_inner_updates": 5,
        "use_armijo_critic": True,
        "use_armijo_actor": True,
        "n_epochs": 10,
        "tau": 0.1,
        "beta": 1.01,
        "cost_limit": 25,
    }
    seed = 2

    train("test_cartpole", "./test-alm-for-plot/", "SPMA-ALM", env_id, seed, algo_params)
    # eg.run(train, num_pool=6)
