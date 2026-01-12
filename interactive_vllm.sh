#!/bin/bash

# Set up log files using SLURM_JOB_ID
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/${SLURM_JOB_ID:-$$}.out"
LOG_ERR="$LOG_DIR/${SLURM_JOB_ID:-$$}.err"

# Redirect all output (including singularity setup noise) to log files
exec > "$LOG_OUT" 2> "$LOG_ERR"

source singularity_launcher.sh

run_sing_bash '
    MODEL=/shared_silo/scratch/models/DeepSeek-V3
    TP=8
    MAX_NUM_SEQS=128
    MAX_NUM_BATCHED_TOKENS=262144
    MAX_MODEL_LEN=32768

    run_python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP \
        --max-model-len $MAX_MODEL_LEN \
        --swap-space 0 \
        --max-num-seqs $MAX_NUM_SEQS \
        --no-enable-prefix-caching \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --block-size 1 \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --quantization fp8
'
