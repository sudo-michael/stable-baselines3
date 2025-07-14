import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np

# def plot_results(log_folder, title="Learning Curve"):
#     """
#     plot the results

#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     df = load_results(log_folder)
#     # plot against timesteps
#     timesteps = np.cumsum(df.l.values)  # type: ignore[arg-type]
#     returns = df.r.values
#     cost_returns = df.c.values
#     # Truncate x
#     timesteps = timesteps[len(timesteps) - len(returns) :]

#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#     plt.suptitle(title)
#     axs[0].plot(timesteps, returns)
#     axs[0].set_xlabel("Timesteps")
#     axs[0].set_ylabel("Returns")

#     axs[1].plot(timesteps, cost_returns)
#     axs[1].set_xlabel("Timesteps")
#     axs[1].set_ylabel("Cost Returns")
#     axs[1].hlines(y=40, xmin=0, xmax=timesteps[-1], label='b', color='black', linestyles='--')
#     plt.savefig('test')
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")




# plot_results('./train/spma_eta_0.3', 'SPMA (eta=0.3)')
etas = [0.3, 0.5, 0.7, 0.9, 1.0]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for eta in etas:
    log_folder = f'./train/spma_eta_{eta}'
    df = load_results(log_folder)

    # plot against timesteps
    timesteps = np.cumsum(df.l.values)  # type: ignore[arg-type]
    returns = moving_average(df.r.values, 50)
    cost_returns = moving_average(df.c.values, 50)
    # Truncate x
    timesteps = timesteps[len(timesteps) - len(returns) :]

    axs[0].plot(timesteps, returns, label=f'eta={eta}')
    axs[0].set_xlabel("Timesteps")
    axs[0].set_ylabel("Returns")
    axs[0].legend()

    axs[1].plot(timesteps, cost_returns, label=f'eta={eta}')
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Cost Returns")
    axs[1].hlines(y=40, xmin=0, xmax=timesteps[-1], label='b', color='black', linestyles='--')


plt.savefig('simple_eta_grid_search')