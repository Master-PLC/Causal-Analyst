#!/bin/bash

gpus=0,3
num_processes=$(echo $gpus | tr "," "\n" | wc -l)

# local_dir="/hub/huggingface/models/Qwen"
local_dir="./ckpts"
model_name="Qwen-7B-Chat"

eval_dir="./eval_results"

CUDA_VISIBLE_DEVICES=$gpus python scripts/vllm_infer.py \
    --model_name_or_path $local_dir/$model_name \
    --template qwen \
    --dataset jailbreak_raw \
    --trust_remote_code true \
    --vllm_config {'gpu_memory_utilization':0.7} \
    --tokenized_path ./tkd_data/jailbreak_raw \
    --save_name "$eval_dir/$model_name/generated_predictions_raw.jsonl"


CUDA_VISIBLE_DEVICES=$gpus python scripts/vllm_infer.py \
    --model_name_or_path $local_dir/$model_name \
    --template qwen \
    --dataset jailbreak_enhanced \
    --vllm_config {'gpu_memory_utilization':0.7} \
    --trust_remote_code true \
    --tokenized_path ./tkd_data/jailbreak_enhanced \
    --save_name "$eval_dir/$model_name/generated_predictions_enhanced.jsonl"
