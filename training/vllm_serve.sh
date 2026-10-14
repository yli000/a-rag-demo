#!/usr/bin/env bash
set -e
# 从仓库根目录运行；模型权重目录和服务名称可通过环境变量覆盖。
vllm serve "${RAG_MODEL_DIR:-output/qwen3_lora_sft_int4}" \
  --served-model-name "${RAG_LLM_MODEL:-qwen3-polo}" \
  --max-model-len 8192 --gpu-memory-utilization 0.75
