#!/bin/bash
#
#SBATCH --job-name=EvalRKMADNI # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH -p sablab-cpu # specify partition
#SBATCH -o ./job_out/%j-eval.out
#SBATCH -e ./job_err/%j-eval.err

source /midtier/sablab/scratch/alm4065/keymorph/.venv/bin/activate

#!/bin/bash

JOB_NAME_PREFIX="eval_ADNI"
NUM_LEVELS_FOR_UNET=5
NUM_KEYPOINTS=64
BATCH_SIZE=1
LEARNING_RATE=3e-6
LOSS_FN="dice"
TRANSFORM_TYPE="tps_uniform"
JOB_NAME="${JOB_NAME_PREFIX}_numlevels${NUM_LEVELS_FOR_UNET}_keypoints${NUM_KEYPOINTS}_batch${BATCH_SIZE}_lr${LEARNING_RATE}_loss${LOSS_FN}_transform${TRANSFORM_TYPE}"

python /midtier/sablab/scratch/alm4065/keymorph/scripts/run.py \
    --run_mode eval \
    --job_name ${JOB_NAME} \
    --align_keypoints_in_real_world_coords \
    --kp_layer com \
    --num_keypoints $NUM_KEYPOINTS \
    --loss_fn $LOSS_FN \
    --transform_type $TRANSFORM_TYPE \
    --train_dataset csv \
    --data_path /midtier/sablab/scratch/alm4065/keymorph/dataset/adni_lowest_10_percent_pairs.csv \
    --save_dir /midtier/sablab/scratch/alm4065/keymorph/experiments/evaluate_ADNI \
    --visualize \
    --use_amp \
    --weighted_kp_align power \
    --backbone truncatedunet \
    --num_levels_for_unet 5 \
    --affine_slope -1 \
    --max_random_affine_augment_params 0 0 0 0 \
    --use_wandb \
    --wandb_kwargs project=keymorph name=${JOB_NAME} dir=/midtier/sablab/scratch/alm4065/wandb/ \
    --epochs 20000 \
    --load_path /midtier/sablab/scratch/alm4065/keymorph/experiments/train_ADNI/__training__train_ADNI_spatial_intensity_numlevels5_keypoints64_batch1_lr3e-6_lossdice_transformtps_uniform/checkpoints/epoch2000_trained_model.pth.tar