#!/bin/bash
#SBATCH -J Atari
#SBATCH --time=0-06:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4%5
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=cc/%x_%a.out

module restore sb3
source ~/sb3/bin/activate

# ---------------------
# Parameters for this task
# ---------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

# ---------------------
# Run experiment
# ---------------------
python train_atari.py \
    --use_wandb False \
    --exp_name atari_scores \
    --slurm_task_id $TASK_ID \
    --seed $TASK_ID \
    --use_objects False


