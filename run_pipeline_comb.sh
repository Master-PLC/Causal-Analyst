#!/bin/bash

gpus=1,2,3,6,7
num_processes=$(echo $gpus | tr "," "\n" | wc -l)


modify_yaml() {
    local yaml_path="$1"
    local save_path="$2"
    shift 2  # 移除前两个参数，剩下的参数是字段更新

    # 构建字段更新参数
    local fields=()
    for field in "$@"; do
        fields+=("$field")
    done

    echo "Calling command: python modify_yaml.py --yaml_path $yaml_path --field ${fields[@]} --save_path $save_path"
    python modify_yaml.py --yaml_path "$yaml_path" --field "${fields[@]}" --save_path "$save_path"
}

config_dir="configs"
cls_yaml_name="qwen2.5-7b-lora-cls"
gl_yaml_name="qwen2.5-7b-lora-gl"
all_yaml_name="qwen2.5-7b-lora-all"

export_yaml_name="qwen2.5-7b-lora-merge"

eval_yaml_name="qwen2.5-7b-lora-eval"
eval_save_name=eval

exp_name=qwen2.5-7b
save_dir="saves/$exp_name"
cls_save_name="cls"
gl_save_name="gl"
all_save_name="all"

export_dir="models/$exp_name"
cls_export_name="cls"
gl_export_name="gl"
all_export_name="all"

model_name_or_path=/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct

re_tokenize=0
if [ "$re_tokenize" -eq 1 ]; then
    echo "Re-tokenizing data"
    rm -rf ./tkd_data/causal

    CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
        --multi_gpu --mixed_precision no --num_processes $num_processes \
        src/train.py "$config_dir/$cls_yaml_name.yaml"
fi

# cls_skip_cycles=(1 2 3 4 5 6 7 8 9 10)
# gl_skip_cycles=(1 2 3 4 5 6 7 8 9 10)
# comb_skip_cycles=(1 2 3 4 5 6 7 8 9 10)
# eval_skip_cycles=(1)
cls_skip_cycles=()
gl_skip_cycles=()
comb_skip_cycles=()
eval_skip_cycles=()

rec_lambda=1.0

lr_cls=0.00005
lr_gl=0.00005
lr_comb=0.00003

train_cycle=10
cls_epoch=2
gl_epoch=2
all_epoch=2

resume_cls=false
resume_gl=false
resume_comb=false


alignment_type="multiplicative"
save_steps=100


TRAIN_START=$(date +%s.%N)

for i in $(seq 1 $train_cycle); do
    echo "Training cycle $i"
    CYC_START=$(date +%s.%N)


    if [ $i -eq 1 ]; then
        first_train=true
    else
        first_train=false
    fi



    echo "Training classifier"
    START=$(date +%s.%N)
    cls_save_dir="$save_dir/$cls_save_name-cyc$i"
    mkdir -p "$cls_save_dir"

    # if cycle i in cls_skip_cycles, skip training
    if [[ " ${cls_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping classifier training for cycle $i"
    else

        modify_yaml "$config_dir/$cls_yaml_name.yaml" "$cls_save_dir/$cls_yaml_name.yaml" \
            "model_name_or_path=$model_name_or_path" \
            "first_train=$first_train" \
            "output_dir=$cls_save_dir" \
            "num_train_epochs=$cls_epoch" \
            "rec_lambda=$rec_lambda" \
            "learning_rate=$lr_cls" \
            "resume_from_checkpoint=$resume_cls" \
            "alignment_type=$alignment_type" \
            "save_steps=$save_steps" \
            "eval_steps=$save_steps" \
            2>&1 | tee "$cls_save_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/train.py "$cls_save_dir/$cls_yaml_name.yaml" \
            2>&1 | tee -a "$cls_save_dir/output.log"

    fi
    END=$(date +%s.%N)
    DUR=$(echo "$END - $START" | bc)
    echo "Training time for classifier: $DUR seconds"



    echo "Exporting classifier"
    cls_export_dir="$export_dir/$cls_export_name-cyc$i"
    mkdir -p "$cls_export_dir"

    # if cycle i in cls_skip_cycles, skip training
    if [[ " ${cls_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping classifier export for cycle $i"
    else
        modify_yaml "$config_dir/$export_yaml_name.yaml" "$cls_export_dir/$export_yaml_name.yaml" \
            "model_name_or_path=$model_name_or_path" \
            "adapter_name_or_path=$cls_save_dir" \
            "export_dir=$cls_export_dir" \
            "additional_target=head_cls" \
            "rec_lambda=$rec_lambda" \
            2>&1 | tee "$cls_export_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/custom_export.py "$cls_export_dir/$export_yaml_name.yaml" \
            2>&1 | tee -a "$cls_export_dir/output.log"
    fi
    model_name_or_path="$cls_export_dir"



    echo "Training graph learner"
    START=$(date +%s.%N)
    gl_save_dir="$save_dir/$gl_save_name-cyc$i"
    mkdir -p "$gl_save_dir"

    # if cycle i in gl_skip_cycles, skip training
    if [[ " ${gl_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping graph learner training for cycle $i"

    else
        modify_yaml "$config_dir/$gl_yaml_name.yaml" "$gl_save_dir/$gl_yaml_name.yaml" \
            "model_name_or_path=$model_name_or_path" \
            "first_train=$first_train" \
            "output_dir=$gl_save_dir" \
            "num_train_epochs=$gl_epoch" \
            "rec_lambda=$rec_lambda" \
            "learning_rate=$lr_gl" \
            "resume_from_checkpoint=$resume_gl" \
            "alignment_type=$alignment_type" \
            "save_steps=$save_steps" \
            "eval_steps=$save_steps" \
            2>&1 | tee "$gl_save_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/train.py "$gl_save_dir/$gl_yaml_name.yaml" \
            2>&1 | tee -a "$gl_save_dir/output.log"

    fi
    first_train=false
    END=$(date +%s.%N)
    DUR=$(echo "$END - $START" | bc)
    echo "Training time for graph learner: $DUR seconds"



    echo "Exporting graph learner"
    gl_export_dir="$export_dir/$gl_export_name-cyc$i"
    mkdir -p "$gl_export_dir"

    # if cycle i in gl_skip_cycles, skip training
    if [[ " ${gl_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping graph learner export for cycle $i"
    else
        modify_yaml "$config_dir/$export_yaml_name.yaml" "$gl_export_dir/$export_yaml_name.yaml" \
            "model_name_or_path=$model_name_or_path" \
            "adapter_name_or_path=$gl_save_dir" \
            "export_dir=$gl_export_dir" \
            "additional_target=head_feat,head_align,graph_learner" \
            "rec_lambda=$rec_lambda" \
            2>&1 | tee "$gl_export_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/custom_export.py "$gl_export_dir/$export_yaml_name.yaml" \
            2>&1 | tee -a "$gl_export_dir/output.log"
    fi
    model_name_or_path="$gl_export_dir"




    echo "Training classifier and graph learner together"
    START=$(date +%s.%N)
    all_save_dir="$save_dir/$all_save_name-cyc$i"
    mkdir -p "$all_save_dir"

    # if cycle i in comb_skip_cycles, skip training
    if [[ " ${comb_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping combination training for cycle $i"

    else
        modify_yaml "$config_dir/$all_yaml_name.yaml" "$all_save_dir/$all_yaml_name.yaml" \
            "do_predict=true" \
            "model_name_or_path=$model_name_or_path" \
            "first_train=$first_train" \
            "output_dir=$all_save_dir" \
            "num_train_epochs=$all_epoch" \
            "rec_lambda=$rec_lambda" \
            "learning_rate=$lr_comb" \
            "resume_from_checkpoint=$resume_comb" \
            "alignment_type=$alignment_type" \
            "save_steps=$save_steps" \
            "eval_steps=$save_steps" \
            2>&1 | tee "$all_save_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/train.py "$all_save_dir/$all_yaml_name.yaml" \
            2>&1 | tee -a "$all_save_dir/output.log"

        resume_comb=false
    fi
    END=$(date +%s.%N)
    DUR=$(echo "$END - $START" | bc)
    echo "Training time for combination: $DUR seconds"



    echo "Exporting combination"
    all_export_dir="$export_dir/$all_export_name-cyc$i"
    mkdir -p "$all_export_dir"
    # if cycle i in comb_skip_cycles, skip training
    if [[ " ${comb_skip_cycles[@]} " =~ " $i " ]]; then
        echo "Skipping combination export for cycle $i"

    else
        modify_yaml "$config_dir/$export_yaml_name.yaml" "$all_export_dir/$export_yaml_name.yaml" \
            "model_name_or_path=$model_name_or_path" \
            "adapter_name_or_path=$all_save_dir" \
            "export_dir=$all_export_dir" \
            "additional_target=head_cls,head_feat,head_align,graph_learner" \
            "rec_lambda=$rec_lambda" \
            2>&1 | tee "$all_export_dir/output.log"

        CUDA_VISIBLE_DEVICES=$gpus accelerate launch \
            --multi_gpu --mixed_precision no --num_processes $num_processes \
            src/custom_export.py "$all_export_dir/$export_yaml_name.yaml" \
            2>&1 | tee -a "$all_export_dir/output.log"
    fi
    model_name_or_path="$all_export_dir"



    # echo "Evaluating classifier and graph learner"
    # START=$(date +%s.%N)
    # result_save_dir="$save_dir/$eval_save_name-cyc$i"
    # mkdir -p "$result_save_dir"

    # # if cycle i in eval_skip_cycles, skip training
    # if [[ " ${eval_skip_cycles[@]} " =~ " $i " ]]; then
    #     echo "Skipping evaluation for cycle $i"

    # else
    #     modify_yaml "$config_dir/$cls_yaml_name.yaml" "$result_save_dir/$eval_yaml_name.yaml" \
    #         "do_train=false" \
    #         "do_predict=true" \
    #         "model_name_or_path=$model_name_or_path" \
    #         "first_train=false" \
    #         "training_stage=3" \
    #         "additional_target=head_cls,head_feat,head_align,graph_learner" \
    #         "output_dir=$result_save_dir" \
    #         "rec_lambda=$rec_lambda" \
    #         "learning_rate=$lr_comb" \
    #         2>&1 | tee "$result_save_dir/output.log"

    #     first_gpu=$(echo $gpus | cut -d ',' -f 1)
    #     CUDA_VISIBLE_DEVICES=$first_gpu accelerate launch \
    #         --mixed_precision no --num_processes 1 \
    #         src/train.py "$result_save_dir/$eval_yaml_name.yaml" \
    #         2>&1 | tee -a "$result_save_dir/output.log"
    # fi
    # END=$(date +%s.%N)
    # DUR=$(echo "$END - $START" | bc)
    # echo "Evaluation time: $DUR seconds"


    CYC_END=$(date +%s.%N)
    CYC_DUR=$(echo "$CYC_END - $CYC_START" | bc)
    echo "Training cycle $i done in $CYC_DUR seconds"
done

TRAIN_END=$(date +%s.%N)
TRAIN_DUR=$(echo "$TRAIN_END - $TRAIN_START" | bc)
echo "Training done in $TRAIN_DUR seconds"




# first_gpu=$(echo $gpus | cut -d ',' -f 1)
# num_processes=1

# eval_yaml_name="qwen2.5-7b-lora-eval"
# eval_save_name=eval

# echo "Evaluating classifier and graph learner"
# START=$(date +%s.%N)
# result_save_dir="$save_dir/$eval_save_name-cyc$train_cycle"
# mkdir -p "$result_save_dir"

# modify_yaml "$config_dir/$cls_yaml_name.yaml" "$result_save_dir/$eval_yaml_name.yaml" \
#     "do_train=false" \
#     "do_predict=true" \
#     "model_name_or_path=$model_name_or_path" \
#     "first_train=false" \
#     "training_stage=3" \
#     "additional_targets=head_cls,head_feat,head_align,graph_learner" \
#     "output_dir=$result_save_dir" \
#     "rec_lambda=$rec_lambda" \
#     2>&1 | tee "$result_save_dir/output.log"

# CUDA_VISIBLE_DEVICES=$first_gpu accelerate launch \
#     --mixed_precision no --num_processes $num_processes \
#     src/train.py "$result_save_dir/$eval_yaml_name.yaml" \
#     2>&1 | tee -a "$result_save_dir/output.log"
# END=$(date +%s.%N)
# DUR=$(echo "$END - $START" | bc)
# echo "Evaluation time: $DUR seconds"


wait