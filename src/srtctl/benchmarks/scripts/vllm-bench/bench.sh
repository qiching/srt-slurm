#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vLLM benchmark using vllm bench serve
# Expects: endpoint isl osl concurrencies [req_rate] [num_prompts_override]
#
# Environment variables for profiling:
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling
#   PROFILE_PREFILL_IPS: Comma-separated list of prefill worker IPs
#   PROFILE_DECODE_IPS: Comma-separated list of decode worker IPs
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP: Step range for prefill
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP: Step range for decode

set -e

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}
NUM_PROMPTS_OVERRIDE=${6:-0}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

MODEL_NAME="${BENCH_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"

echo "vLLM-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}"

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Start profiling
start_all_profiling

# Run benchmark for each concurrency level
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    if [ "$NUM_PROMPTS_OVERRIDE" -gt 0 ] 2>/dev/null; then
        num_prompts=$NUM_PROMPTS_OVERRIDE
    else
        num_prompts=$((concurrency * 10))
    fi

    echo "Running benchmark with concurrency: $concurrency, num_prompts: $num_prompts"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    set -x
    vllm bench serve \
        --model "${MODEL_NAME}" \
        --host "${HOST}" --port "${PORT}" \
        --dataset-name random \
        --max-concurrency "${concurrency}" \
        --num-prompts "${num_prompts}" \
        --input-len "${ISL}" \
        --output-len "${OSL}" \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --trust-remote-code
    set +x

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

stop_all_profiling

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "vLLM-Bench completed"
