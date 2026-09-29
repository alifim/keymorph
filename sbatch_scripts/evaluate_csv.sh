#!/bin/bash
#
#SBATCH --job-name=EvalRKM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume?
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p sablab-gpu # specify partition
#SBATCH -o ./job_out/%j-eval.out
#SBATCH -e ./job_err/%j-eval.err

set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: sbatch $0 /path/to/checkpoint.pth.tar {adni|abdominal}" >&2
    exit 2
fi

ROOT="/midtier/sablab/scratch/alm4065/keymorph"
LOAD_PATH="$1"
case "${2,,}" in
    adni) DATASET="ADNI"; DATA_PATH="$ROOT/dataset/adni_lowest_10_percent_pairs.csv" ;;
    abdominal) DATASET="AbdominalMRI"; DATA_PATH="$ROOT/dataset/new_data_pairs_preprocessed.csv" ;;
    *) echo "Dataset must be adni or abdominal" >&2; exit 2 ;;
esac
TRAIN_DIR="$(dirname "$(dirname "$LOAD_PATH")")"
TRAIN_ARGS="$TRAIN_DIR/args.json"

if [[ ! -f "$LOAD_PATH" || ! -f "$TRAIN_ARGS" || ! -f "$DATA_PATH" ]]; then
    echo "Missing checkpoint, training args.json, or data CSV: $LOAD_PATH / $TRAIN_ARGS / $DATA_PATH" >&2
    exit 1
fi

source "$ROOT/.venv/bin/activate"

# Read the training settings and emit one command-line argument per line.
metadata=$(python - "$ROOT" "$TRAIN_ARGS" "$DATASET" <<'PY'
import json
import sys

root, args_file, dataset = sys.argv[1:]
with open(args_file) as f:
    saved = json.load(f)

print("--save_dir")
print(f"{root}/experiments/evaluate_RKM_{dataset}_{saved['num_keypoints']}keypoints")

for key in (
    "num_keypoints", "num_levels_for_unet", "num_truncated_layers_for_truncatedunet",
    "backbone", "kp_layer", "norm_type", "dim", "loss_fn", "transform_type",
    "lr", "lambda_mask", "lambda_dispersion", "loss_alpha", "max_train_keypoints",
    "num_subgrids", "num_resolutions_for_itkelastix", "seed", "aug_strategy",
    "num_test_subjects", "weighted_kp_align",
):
    value = saved[key]
    if value is not None:
        if "\n" in str(value):
            raise SystemExit(f"Invalid newline in {key}")
        print(f"--{key}")
        print(value)

for key in (
    "align_keypoints_in_real_world_coords", "use_amp", "use_checkpoint",
    "compute_subgrids_for_tps", "mix_modalities",
):
    if saved[key]:
        print(f"--{key}")
PY
)
mapfile -t model_flags <<< "$metadata"

TRAIN_NAME="${TRAIN_DIR##*/}"
TRAIN_NAME="${TRAIN_NAME#__training__}"
CHECKPOINT_NAME="${LOAD_PATH##*/}"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.pth.tar}"
JOB_NAME="eval_RKM_${DATASET}_${TRAIN_NAME}_${CHECKPOINT_NAME}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

python "$ROOT/scripts/run.py" \
    --run_mode eval \
    --job_name "$JOB_NAME" \
    --load_path "$LOAD_PATH" \
    --train_dataset csv \
    --data_path "$DATA_PATH" \
    --batch_size 1 \
    --visualize \
    "${model_flags[@]}"
