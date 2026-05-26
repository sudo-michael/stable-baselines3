import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the evaluations file
log_dir = "./runs/test/task_id_1/"
data = np.load(f"{log_dir}/evaluations.npz")
print(data)
print(data['timesteps'])
results = data['results']
mean_reward = results.mean(axis=1)
std_reward = results.std(axis=1)
print(mean_reward, std_reward)
