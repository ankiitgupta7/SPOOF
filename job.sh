#!/bin/bash --login
#SBATCH --job-name=few_pixel_attack
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:59:00
#SBATCH --array=0-179

# === Derived from SLURM_ARRAY_TASK_ID ===
MODELS=("VGG16" "ResNet-50" "ViT-B_16")
SEEDS=(4201 4202 4203)
BATCH_INDEX=$(( SLURM_ARRAY_TASK_ID % 20 ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID / 20) % 3 ))
MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID / 60 ))

MODEL=${MODELS[$MODEL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
START_SAMPLE=$(( BATCH_INDEX * 50 ))
END_SAMPLE=$(( START_SAMPLE + 50 ))

echo "Running model=$MODEL, seed=$SEED, samples=$START_SAMPLE to $(($END_SAMPLE - 1))"


# Ensure required directories exist
mkdir -p logs

# Initialize Conda
source /mnt/ufs18/home-253/guptaa23/miniforge3/etc/profile.d/conda.sh
conda activate SPOOF
export PATH="/mnt/ufs18/home-253/guptaa23/miniforge3/envs/SPOOF/bin:$PATH"


for (( SAMPLE=$START_SAMPLE; SAMPLE<$END_SAMPLE; SAMPLE++ ))
do
    python fpa_test.py --model "$MODEL" --seed "$SEED" --sample "$SAMPLE"
done
