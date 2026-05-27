#!/bin/bash
#SBATCH -J Atari
#SBATCH --time=0-12:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14%15
#SBATCH --output=cc/%x_%a.out

module restore sb3
source ~/sb3/bin/activate

# ---------------------
# Parameters for this task
# ---------------------
PARAMS=("30 0"  "30 1" "30 2" "30 3" "30 4" "40 0"  "40 1" "40 2" "40 3" "40 4" "50 0"  "50 1" "50 2" "50 3" "50 4" )

TASK_ID=${SLURM_ARRAY_TASK_ID}
PARAM="${PARAMS[$TASK_ID]}"
read boundary_y seed <<< "$PARAM"

# ---------------------
# Run experiment
# ---------------------
python train_atari.py \
    --use_wandb False \
    --exp_name atari_scores_changing_y_boundary \
    --slurm_task_id $TASK_ID \
    --seed $seed \
    --use_objects False \
    --pong_boundary_y $boundary_y \


