# %%
import glob
import os
import scipy
from matplotlib import pyplot as plt
import numpy as np
from tbparse import SummaryReader
import pandas as pd

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
# %%
seed = 0
log_path  ="./test-alm-for-plot"
eval_files = glob.glob(os.path.join(log_path, "*npz"))
eval_logs = [np.load(eval_file) for eval_file in eval_files]
# %%
num_inner_updates = 5


fig, axs = plt.subplots(1, 1, figsize=(12, 5))
timesteps = eval_logs[0]["timesteps"]
returns = np.array([log["results"] for log in eval_logs]).mean(axis=2)
# index to only plot results after the end of the inner-loop
timesteps = timesteps[num_inner_updates-1::num_inner_updates]
returns = returns[:,num_inner_updates-1::num_inner_updates]
mean_returns, mean_returns_ci = mean_confidence_interval(returns)
axs.plot(timesteps, mean_returns)
axs.fill_between(timesteps, mean_returns - mean_returns_ci, mean_returns + mean_returns_ci, alpha=0.4, linewidth=0)
axs.set_xlabel("Timesteps")
axs.set_ylabel("Returns")

# %%
logs = glob.glob(os.path.join(log_path, "*csv"))
logs_df = [pd.read_csv(log) for log in logs]
# %%
timesteps = logs_df[0]['time/total_timesteps'].to_numpy()

rollout_ep_mean = [df['rollout/ep_rew_mean'] for df in logs_df]

fig, axs = plt.subplots(1, 1, figsize=(12, 5))
mean_returns, mean_returns_ci = mean_confidence_interval(rollout_ep_mean)
axs.plot(timesteps, mean_returns)
axs.fill_between(timesteps, mean_returns - mean_returns_ci, mean_returns + mean_returns_ci, alpha=0.4, linewidth=0)
axs.set_xlabel("Timesteps")
axs.set_ylabel("Returns")
# %%
