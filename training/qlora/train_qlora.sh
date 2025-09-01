#!/bin/bash

RESULT_DIR="./training/qlora/results"

echo "Base output: $RESULT_DIR"
echo ""

# Define models to test
declare -a MODELS=(
    # "Qwen/Qwen3-4B"
    # "google/gemma-3-4b-it"
    # "CohereLabs/c4ai-command-r7b-12-2024"
    # mistralai/Mistral-7B-Instruct-v0.3
    # "speakleash/Bielik-4.5B-v3.0-Instruct"
    "speakleash/Bielik-1.5B-v3.0-Instruct"
    # "Qwen/Qwen3-1.7B"
    "meta-llama/Llama-3.2-3B-Instruct"
    # "google/gemma-3-1b-it"
)

# Define configurations to test
# params: name r alpha dropout epochs lr grad_steps
declare -a CONFIGS=(
    "conservative:2:4:0.3:1:5e-5:8"
    "moderate:8:16:0.25:2:5e-5:8"  
    "agressive:16:32:0.2:3:1e-4:6"
)


echo "✅ Found ${#MODELS[@]} models and ${#CONFIGS[@]} configurations to test"
echo "Total experiments: $((${#MODELS[@]} * ${#CONFIGS[@]}))"
echo ""

# Results tracking
declare -a ALL_RESULTS=()

# Train and evaluate each model with each configuration
for MODEL in "${MODELS[@]}"; do
    # Create model-specific directory name
    MODEL_DIR=$(echo "$MODEL" | sed 's|/|_|g' | tr '[:upper:]' '[:lower:]')
    
    echo "Testing Model: $MODEL"
    echo "======================================="
    echo ""
    
    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r name r alpha dropout epochs lr grad_steps <<< "$config"
        
        EXPERIMENT_DIR="${RESULT_DIR}/${MODEL_DIR}/${name}"
        
        echo "🔧 Training: ${MODEL_DIR}/${name}"
        echo "   Config: r=$r, α=$alpha, lr=$lr"
        
        # Train this model+config combination
        python3 training/qlora/train_models.py \
            --model_name "$MODEL" \
            --output_dir "$EXPERIMENT_DIR" \
            --lora_r $r \
            --lora_alpha $alpha \
            --lora_dropout $dropout \
            --epochs $epochs \
            --learning_rate $lr \
            --gradient_accumulation_steps $grad_steps \
            --batch_size 1
        
        if [ $? -eq 0 ]; then
            echo "{MODEL_DIR}/${name} training completed"
        else
            echo "❌ ${MODEL_DIR}/${name} training failed"
        fi    
    done
    
    echo ""
done

