#!/bin/bash

uv run train_atari.py --exp_name "atari_pong_eval_object" --seed 0 --slurm_id 0
uv run train_atari.py --exp_name "atari_pong_eval_object" --seed 1 --slurm_id 1
uv run train_atari.py --exp_name "atari_pong_eval_object" --seed 2 --slurm_id 2
uv run train_atari.py --exp_name "atari_pong_eval_object" --seed 3 --slurm_id 3
uv run train_atari.py --exp_name "atari_pong_eval_object" --seed 4 --slurm_id 4

