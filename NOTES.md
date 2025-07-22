# How to determine best hyper-parameters
a. select highest reward that satisfies cost rate?

# How to estimate $V^{\pi}_c(\rho)$?
For a brief overview, the training loop in SB3 is that trajectories are collected until a target
number of samples are collected. A trajectory will be terminated early if the target is reached.

Given the collected samples in the rollout buffer, here are the ways of estimating $V^{\pi}_c(\rho)$
1. Use the emperical undiscounted cost: $1/N \sum_{i=1}^N \sum_{t=0}^T c(s^i_t, a^i_t)$


# Installation
- TODO
