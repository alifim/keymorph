#!/bin/bash
#
#SBATCH --job-name=TrainRKMAbdominal # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-train.out
#SBATCH -e ./job_err/%j-train.err

source /midtier/sablab/scratch/alm4065/keymorph/.venv/bin/activate

#!/bin/bash

JOB_NAME_PREFIX="train_CT"
NUM_LEVELS_FOR_UNET=5
NUM_KEYPOINTS=64
BATCH_SIZE=1
LEARNING_RATE=3e-6
LOSS_FN="dice+ssim"
TRANSFORM_TYPE="tps_uniform"
JOB_NAME="${JOB_NAME_PREFIX}_numlevels${NUM_LEVELS_FOR_UNET}_keypoints${NUM_KEYPOINTS}_batch${BATCH_SIZE}_lr${LEARNING_RATE}_loss${LOSS_FN}_transform${TRANSFORM_TYPE}"

#JOB_NAME="some_testing"
python /midtier/sablab/scratch/alm4065/keymorph/scripts/run.py \
    --run_mode train \
    --job_name ${JOB_NAME} \
    --align_keypoints_in_real_world_coords \
    --kp_layer com \
    --num_keypoints ${NUM_KEYPOINTS} \
    --loss_fn ${LOSS_FN} \
    --transform_type ${TRANSFORM_TYPE} \
    --train_dataset csv \
    --data_path /midtier/sablab/scratch/alm4065/keymorph/dataset/AbdomenCTCT_30_samples_combination.csv \
    --save_dir /midtier/sablab/scratch/alm4065/keymorph/experiments/training_real_world_coordinates_CT \
    --visualize \
    --use_amp \
    --weighted_kp_align power \
    --backbone truncatedunet \
    --num_levels_for_unet ${NUM_LEVELS_FOR_UNET} \
    --affine_slope -1 \
    --max_random_affine_augment_params 0 0 0 0 \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/alm4065/wandb/ \
    --epochs 20000 \
    --load_path /midtier/sablab/scratch/omt4002/keymorph/expriments/pretraining_real_world_coordinates_CT/__pretrain__pretrain_CT_numlevels5_same_mod_training_64_power_weighted_keypoints64_batch1_lr3e-06/checkpoints/pretrained_epoch14475_model.pth.tar