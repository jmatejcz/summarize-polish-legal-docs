#!/bin/bash

RESULT_DIR="./training/prompt_tuning/results"
echo "Base output: $RESULT_DIR"
echo ""

# Define models to test
MODELS=(
    "google/gemma-3-4b-it"
    "google/gemma-3-1b-it"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-1.7B"
    "speakleash/Bielik-4.5B-v3.0-Instruct"
    "speakleash/Bielik-1.5B-v3.0-Instruct"
    "CohereLabs/c4ai-command-r7b-12-2024"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.2-3B-Instruct"
)

# Define configurations to test
# params: name:num_virtual_tokens:epochs:learning_rate:gradient_accumulation_steps:batch_size
declare -a CONFIGS=(
    "conservative:10:4:0.1:8:1"
    "moderate:20:5:0.12:6:2"
    "aggressive:30:6:0.15:4:2"
)

echo "✅ Found ${#MODELS[@]} models and ${#CONFIGS[@]} configurations to test"
echo "Total experiments: $((${#MODELS[@]} * ${#CONFIGS[@]}))"
echo ""

# Results tracking
declare -a ALL_RESULTS=()

# Train each model with each configuration
for MODEL in "${MODELS[@]}"; do
    # Create model-specific directory name
    MODEL_DIR=$(echo "$MODEL" | sed 's|/|_|g' | tr '[:upper:]' '[:lower:]')
    
    echo "Testing Model: $MODEL"
    echo "======================================="
    echo ""
    
    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r name num_tokens epochs lr grad_steps batch_size val_split <<< "$config"
        
        EXPERIMENT_DIR="${RESULT_DIR}/${MODEL_DIR}/${name}"
        
        echo "🔧 Training: ${MODEL_DIR}/${name}"
        echo " Config: tokens=$num_tokens, epochs=$epochs, lr=$lr"
        echo " Batch size=$batch_size, grad_steps=$grad_steps, val_split=$val_split"
        
        # Train this model+config combination
        python3 training/prompt_tuning/train_models.py \
            --model_name "$MODEL" \
            --output_dir "$EXPERIMENT_DIR" \
            --num_virtual_tokens $num_tokens \
            --epochs $epochs \
            --learning_rate $lr \
            --batch_size 1 \
            --gradient_accumulation_steps $grad_steps \
        
        if [ $? -eq 0 ]; then
            echo "✅ ${MODEL_DIR}/${name} training completed"
            ALL_RESULTS+=("${MODEL_DIR}/${name}: SUCCESS")
        else
            echo "❌ ${MODEL_DIR}/${name} training failed"
            ALL_RESULTS+=("${MODEL_DIR}/${name}: FAILED")
        fi
        
        echo ""
    done
    
    echo ""
done

