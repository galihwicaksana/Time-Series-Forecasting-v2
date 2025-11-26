#!/bin/bash

# Run LSTM training for ETTh1 dataset
# This script trains an LSTM model on the ETTh1 time series dataset

echo "=================================="
echo "LSTM Training for ETTh1 Dataset"
echo "=================================="

# Check if required directories exist
if [ ! -d "dataset/ETT-small" ]; then
    echo "Error: dataset/ETT-small directory not found"
    exit 1
fi

if [ ! -f "dataset/ETT-small/ETTh1.csv" ]; then
    echo "Error: ETTh1.csv not found in dataset/ETT-small/"
    exit 1
fi

# Check if LSTM model directory exists
if [ ! -d "LSTM-Neural-Network-for-Time-Series-Prediction" ]; then
    echo "Error: LSTM-Neural-Network-for-Time-Series-Prediction directory not found"
    exit 1
fi

echo "Starting LSTM training..."
echo ""

# Run the training script
python run_lstm_etth1.py

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
