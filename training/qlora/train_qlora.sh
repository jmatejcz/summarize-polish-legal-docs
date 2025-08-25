#!/bin/bash

# Simple multi-model, multi-config training and evaluation pipeline
# Usage: ./run_pipeline.sh

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="./results_${TIMESTAMP}"

echo "Base output: $BASE_DIR"
echo ""

# Define models to test
declare -a MODELS=(
    # "Qwen/Qwen3-4B"
    "google/gemma-3-4b-it"
    # "speakleash/Bielik-4.5B-v3.0-Instruct"
)

# Define configurations to test
declare -a CONFIGS=(
    "ultra_conservative:2:4:0.3:1:5e-5:8"
    "very_conservative:4:8:0.25:2:5e-5:8"  
    "conservative:8:16:0.2:3:1e-4:6"
    "moderate:12:24:0.15:2:5e-5:6"
)

# Check required files
if [ ! -f "summarize/qlora/train_models.py" ] || [ ! -f "summarize/qlora/test_models.py" ]; then
    echo "❌ Required scripts not found"
    exit 1
fi

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
        
        EXPERIMENT_DIR="${BASE_DIR}/${MODEL_DIR}/${name}"
        
        echo "🔧 Training: ${MODEL_DIR}/${name}"
        echo "   Config: r=$r, α=$alpha, lr=$lr"
        
        # Train this model+config combination
        python3 summarize/qlora/train_models.py \
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
            
            # Evaluate immediately
            echo "Evaluating ${MODEL_DIR}/${name}..."
            echo "experiment dir: ${EXPERIMENT_DIR}"
            
            
            python3 summarize/qlora/test_models.py \
                --training_output_dir "${BASE_DIR}/${MODEL_DIR}" \
                --model_name "$MODEL" > "${MODEL_DIR}_${name}_results.tmp" 2>/dev/null
        else
            echo "❌ ${MODEL_DIR}/${name} training failed"
        fi    
    done
    
    echo ""
done

