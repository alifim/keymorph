#!/bin/bash
#
#SBATCH --job-name=train # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=150:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH --exclude=ai-gpu06,ai-gpu08 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-train.out
#SBATCH -e ./job_err/%j-train.err

source /home/omt4002/miniconda3/bin/activate keymorph-env

#!/bin/bash

JOB_NAME="eval_CT_numlevels5_same_mod_training_128_dice_loss_power_weighted_tps_uniform"
#JOB_NAME="some_testing"
python /home/omt4002/keymorph/scripts/run.py \
    --run_mode eval \
    --job_name ${JOB_NAME} \
    --align_keypoints_in_real_world_coords \
    --kp_layer com \
    --num_keypoints 128 \
    --loss_fn ssim \
    --transform_type tps_uniform \
    --train_dataset csv \
    --data_path /home/omt4002/keymorph/abdominal_scripts/AbdomenCTCT_30_samples_combination.csv \
    --save_dir /midtier/sablab/scratch/omt4002/keymorph/expriments/evaluate_real_world_coordinates_CT \
    --visualize \
    --use_amp \
    --weighted_kp_align power \
    --backbone truncatedunet \
    --num_levels_for_unet 5 \
    --affine_slope -1 \
    --max_random_affine_augment_params 0 0 0 0 \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/omt4002/wandb/ \
    --epochs 20000 \
    --load_path /midtier/sablab/scratch/omt4002/keymorph/expriments/training_real_world_coordinates_CT/__training__train_CT_numlevels5_same_mod_training_128_power_weighted_TPS_keypoints128_batch1_lr3e-06/checkpoints/epoch2000_trained_model.pth.tar