#!/bin/bash
#
#SBATCH --job-name=PreRKMAbdominal # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH --gres=gpu:a100:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-pretrain.out
#SBATCH -e ./job_err/%j-pretrain.err


source /midtier/sablab/scratch/alm4065/keymorph/.venv/bin/activate
#!/bin/bash

JOB_NAME="pretrain_CT_numlevels5_same_mod_training_128_power_weighted" # i was here :)
#JOB_NAME="debug"
python /midtier/sablab/scratch/alm4065/keymorph/scripts/run.py \
    --run_mode pretrain \
    --job_name ${JOB_NAME} \
    --align_keypoints_in_real_world_coords \
    --kp_layer com \
    --num_keypoints 128 \
    --loss_fn mse \
    --transform_type affine \
    --train_dataset csv \
    --data_path /midtier/sablab/scratch/alm4065/keymorph/dataset/AbdomenCTCT_30_samples_combination.csv \
    --save_dir /midtier/sablab/scratch/alm4065/keymorph/experiments/pretraining_real_world_coordinates_CT \
    --visualize \
    --use_amp \
    --weighted_kp_align power \
    --backbone truncatedunet \
    --num_levels_for_unet 5 \
    --affine_slope -1 \
    --max_random_affine_augment_params 0. 0. 0. 0. \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/alm4065/wandb/ \
    --epochs 15000 \
