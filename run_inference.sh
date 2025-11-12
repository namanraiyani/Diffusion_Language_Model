set -e

CHECKPOINT_PATH="${1:-.}"
PROMPT="${2:-}"
SEQ_LEN="${3:-256}"
NUM_STEPS="${4:-256}"
STRATEGY="${5:-random}"
DEVICE="${6:-cuda}"
HF_MODEL="answerdotai/ModernBERT-base"

echo "Activating environment..."
conda activate mlm_env 2>/dev/null || echo "Please activate your conda environment"

pip install -q -r requirements.txt

if [ ! -f "$CHECKPOINT_PATH" ] && [ "$CHECKPOINT_PATH" != "." ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Starting inference..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Sequence Length: $SEQ_LEN"
echo "Number of Steps: $NUM_STEPS"
echo "Strategy: $STRATEGY"
echo "Device: $DEVICE"

if [ -z "$PROMPT" ]; then
    echo "Mode: Unconditional Generation"
    python inference.py \
        --safetensors_path "$CHECKPOINT_PATH" \
        --hf_model_name "$HF_MODEL" \
        --seq_len "$SEQ_LEN" \
        --num_steps "$NUM_STEPS" \
        --strategy "$STRATEGY" \
        --device "$DEVICE"
else
    echo "Mode: Conditional Generation (Q&A)"
    echo "Prompt: $PROMPT"
    python inference.py \
        --safetensors_path "$CHECKPOINT_PATH" \
        --hf_model_name "$HF_MODEL" \
        --prompt "$PROMPT" \
        --seq_len "$SEQ_LEN" \
        --num_steps "$NUM_STEPS" \
        --strategy "$STRATEGY" \
        --device "$DEVICE"
fi

echo "Inference complete!"