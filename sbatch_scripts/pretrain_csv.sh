#!/bin/bash
#
#SBATCH --job-name=PreRKMAbdominal # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-pretrain.out
#SBATCH -e ./job_err/%j-pretrain.err

source /midtier/sablab/scratch/alm4065/keymorph/.venv/bin/activate

# --- Parameters ---
JOB_NAME_PREFIX="pretrain_AbdominalMRI"
NUM_LEVELS_FOR_UNET=5
NUM_KEYPOINTS=64
LOSS_FN="mse"
TRANSFORM_TYPE="affine"
WEIGHTED_KP_ALIGN="power"
TRAINING_MODE="same_mod"
LEARNING_RATE=1e-3

# Construct job name dynamically (matches your original job name string)
JOB_NAME="${JOB_NAME_PREFIX}_numlevels${NUM_LEVELS_FOR_UNET}_${TRAINING_MODE}_training_${NUM_KEYPOINTS}_${WEIGHTED_KP_ALIGN}_weighted_lr${LEARNING_RATE}_loss${LOSS_FN}_transform${TRANSFORM_TYPE}"

python /midtier/sablab/scratch/alm4065/keymorph/scripts/run.py \
    --run_mode pretrain \
    --job_name ${JOB_NAME} \
    --align_keypoints_in_real_world_coords \
    --kp_layer com \
    --num_keypoints $NUM_KEYPOINTS \
    --loss_fn $LOSS_FN \
    --transform_type $TRANSFORM_TYPE \
    --train_dataset csv \
    --data_path /midtier/sablab/scratch/alm4065/keymorph/dataset/new_data_pairs_preprocessed.csv \
    --save_dir /midtier/sablab/scratch/alm4065/keymorph/experiments/pretrain_AbdominalMRI \
    --visualize \
    --use_amp \
    --weighted_kp_align $WEIGHTED_KP_ALIGN \
    --backbone truncatedunet \
    --num_levels_for_unet $NUM_LEVELS_FOR_UNET \
    --affine_slope -1 \
    --max_random_affine_augment_params 0. 0. 0. 0. \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/alm4065/wandb/ \
    --lr $LEARNING_RATE \
    --epochs 15000
