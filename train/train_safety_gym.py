import gymnasium
import safety_gymnasium
from stable_baselines3.common.callbacks import CostEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.spma_pd.spma_pd import SPMAPD

exp_log_dir = "safe_hopper_velocity_spma_pd"

seed = 1
set_random_seed(seed, using_cuda=True)


def make_safe_env(env_id):
    safe_env = safety_gymnasium.make(env_id)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)
    return env


def env_lambda():
    return make_safe_env("SafetyHopperVelocity-v1")


algo_params = {
    "eta": 0.3,
    "n_steps": 2048,
    "batch_size": 2048,
    "total_timesteps": 2048 * 500,
    "num_inner_updates": 1,
    "use_armijo_critic": True,
    "use_armijo_actor": True,
    "n_epochs": 10,
    "lambda_lr": 0.1,
    "cost_limit": 25,
}


env = make_vec_env(env_lambda, seed=seed)
eval_env = make_vec_env(env_lambda, seed=seed + 1337)

total_timesteps = algo_params.pop("total_timesteps")
num_inner_updates = algo_params.pop("num_inner_updates")

callback = CostEvalCallback(
    eval_env,
    eval_freq=algo_params["n_steps"] * num_inner_updates + 1,
    log_path=exp_log_dir,
    deterministic=False,
    verbose=0,
)
model = SPMAPD("MlpPolicy", env, verbose=1, seed=seed, device="cuda", **algo_params, tensorboard_log="spma_pd_tensorboard")
model.learn(total_timesteps=total_timesteps + 1, num_inner_updates=num_inner_updates, progress_bar=True, callback=callback)
