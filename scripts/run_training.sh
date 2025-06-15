#!/bin/bash

# Activate the virtual environment if needed
# source venv/bin/activate

# Set variables for data and model paths
RAW_DATA_PATH="data/raw"
PROCESSED_DATA_PATH="data/processed"
MODEL_OUTPUT_PATH="models/phi2_finetuned"

# Create output directory if it doesn't exist
mkdir -p $MODEL_OUTPUT_PATH

# Run the training script
python src/train.py --raw_data $RAW_DATA_PATH --processed_data $PROCESSED_DATA_PATH --model_output $MODEL_OUTPUT_PATH

# Optionally, you can add commands to evaluate the model after training
# python src/evaluate.py --model_path $MODEL_OUTPUT_PATH

# Optionally, you can add commands to export the model to ONNX format
# python src/export_onnx.py --model_path $MODEL_OUTPUT_PATH --onnx_output_path "models/phi2_finetuned.onnx"