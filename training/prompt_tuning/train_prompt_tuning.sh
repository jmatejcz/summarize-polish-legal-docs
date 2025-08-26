#!/bin/bash

echo "Starting Prompt Tuning Experiments..."
echo "=================================="

# Base parameters
BASE_SCRIPT="summarize/prompt_tuning/training.py"
BASE_OUTPUT_DIR="summarize/prompt_tuning/results"
VAL_SPLIT=0.2
BATCH_SIZE=1

# Define models to test
MODELS=(
    "Qwen/Qwen3-4B"
    "speakleash/Bielik-4.5B-v3.0-Instruct"
)

# Define experiment configurations
# Format: "name:num_virtual_tokens:epochs:learning_rate:gradient_accumulation_steps:use_quantization"
CONFIGS=(
    "conservative:10:4:0.1:8:true"
    "moderate:20:5:0.12:6:true"
    "aggressive:30:6:0.15:4:true"
)

# Function to extract model name for directory
get_model_dir_name() {
    local model=$1
    if [[ $model == *"Qwen3-4B"* ]]; then
        echo "qwen3-4b"
    elif [[ $model == *"Bielik"* ]]; then
        echo "bielik-4.5b"
    else
        echo "unknown"
    fi
}

experiment_num=1

for model in "${MODELS[@]}"; do
    model_dir_name=$(get_model_dir_name "$model")
    
    echo "Testing model: $model"
    echo "===================="
    
    for config in "${CONFIGS[@]}"; do
        # Parse configuration
        IFS=':' read -r config_name num_tokens epochs lr grad_steps use_quant <<< "$config"
        
        # Create output directory name
        output_dir="$BASE_OUTPUT_DIR/${model_dir_name}-${config_name}"
        
        echo "Experiment $experiment_num: $(basename $model) - $config_name"
        echo "Config: tokens=$num_tokens, epochs=$epochs, lr=$lr, grad_steps=$grad_steps, quant=$use_quant"
        
        cmd="python $BASE_SCRIPT \
            --model_name \"$model\" \
            --output_dir \"$output_dir\" \
            --num_virtual_tokens $num_tokens \
            --epochs $epochs \
            --learning_rate $lr \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $grad_steps \
            --val_split $VAL_SPLIT"
        
        if [ "$use_quant" = "true" ]; then
            cmd="$cmd --use_quantization"
        fi
        
        eval $cmd
        
        echo "Experiment $experiment_num completed!"
        echo "Output saved to: $output_dir"
        echo "--------------------"
        
        ((experiment_num++))
    done
    
    echo ""
done

echo "All experiments completed successfully!"
echo "======================================"
echo "Total experiments run: $((experiment_num - 1))"
echo ""
echo "Results saved in $BASE_OUTPUT_DIR/ with the following structure:"
echo "- qwen3-4b-conservative, qwen3-4b-moderate, qwen3-4b-aggressive, etc."
echo "- bielik-4.5b-conservative, bielik-4.5b-moderate, bielik-4.5b-aggressive, etc."
echo ""
echo "Each model was tested with the same parameter configurations:"
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r config_name num_tokens epochs lr grad_steps use_quant <<< "$config"
    echo "  - $config_name: $num_tokens tokens, $epochs epochs, LR $lr, $grad_steps grad steps, quant=$use_quant"
done