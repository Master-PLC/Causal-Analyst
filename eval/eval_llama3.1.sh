#!/bin/bash

gpus=0,1,5
num_processes=$(echo $gpus | tr "," "\n" | wc -l)

local_dir="/hub/huggingface/models/Llama"
model_name="LLaMA-3.1-8B-Instruct"

eval_dir="./eval_results"

CUDA_VISIBLE_DEVICES=$gpus python scripts/vllm_infer.py \
    --model_name_or_path $local_dir/$model_name \
    --template llama3 \
    --dataset jailbreak_raw \
    --save_name "$eval_dir/$model_name/generated_predictions_raw.jsonl"


CUDA_VISIBLE_DEVICES=$gpus python scripts/vllm_infer.py \
    --model_name_or_path $local_dir/$model_name \
    --template llama3 \
    --dataset jailbreak_enhanced \
    --save_name "$eval_dir/$model_name/generated_predictions_enhanced.jsonl"