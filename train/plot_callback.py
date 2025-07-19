import numpy as np
import os
import matplotlib.pyplot as plt
import glob
path = "spma_alm_2/"
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
eta = 0.1
tau=0.1# plot against timesteps
eval_files = glob.glob(os.path.join(path, "*npz"))
logs = np.load(eval_files[0])
timesteps = logs["timesteps"]
returns = np.array(logs["results"]).mean(axis=1)
costs = np.array(logs["costs"]).mean(axis=1)
lambdas = np.array(logs["lambda"])

axs[0].plot(timesteps, returns, label=f"{eta=}-{tau=}")
axs[0].set_xlabel("Timesteps")
axs[0].set_ylabel("Returns")
axs[0].legend()

axs[1].plot(timesteps, costs, label=f"{eta=}-{tau=}")
axs[1].set_xlabel("Timesteps")
axs[1].set_ylabel("Cost Returns")
axs[1].hlines(y=25, xmin=0, xmax=timesteps[-1], label="b", color="black", linestyles="--")

axs[2].plot(timesteps, lambdas, label=f"{eta=}-{tau=}")
axs[2].set_xlabel("Timesteps")
axs[2].set_ylabel("Lambda")

plt.savefig(f"SPMA-ALM-new")
