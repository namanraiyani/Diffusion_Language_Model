set -e

EXPERIMENT_NAME="mlm_experiment_$(date +%Y%m%d_%H%M%S)"
WORKING_DIR="./experiments"
MODE="${1:-pretrain}"  
HF_MODEL="answerdotai/ModernBERT-base"

mkdir -p "$WORKING_DIR"
mkdir -p "./data/raw"
mkdir -p "./data/processed"

echo "Setting up environment..."
conda activate mlm_env 2>/dev/null || echo "Please activate your conda environment"

pip install -q -r requirements.txt

echo "Starting training in $MODE mode..."
echo "Experiment: $EXPERIMENT_NAME"

if [ "$MODE" = "pretrain" ]; then
    python train.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --working_directory "$WORKING_DIR" \
        --mode "pretrain" \
        --hf_model_name "$HF_MODEL" \
        --context_length 512 \
        --per_gpu_batch_size 8 \
        --num_training_steps 1000 \
        --learning_rate 5e-5 \
        --evaluation_interval 500 \
        --checkpoint_interval 500 \
        --max_samples  20000 \
        --logging_steps 10
        
elif [ "$MODE" = "sft" ]; then
    PRETRAINED_CHECKPOINT="${2:-./experiments/latest_model.safetensors}"
    python train.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --working_directory "$WORKING_DIR" \
        --mode "sft" \
        --hf_model_name "$HF_MODEL" \
        --path_to_pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
        --context_length 512 \
        --per_gpu_batch_size 8 \
        --num_training_steps 500 \
        --learning_rate 1e-5 \
        --evaluation_interval 250 \
        --checkpoint_interval 250
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_training.sh [pretrain|sft] [checkpoint_path_for_sft]"
    exit 1
fi

echo "Training complete! Results saved to $WORKING_DIR/$EXPERIMENT_NAME"