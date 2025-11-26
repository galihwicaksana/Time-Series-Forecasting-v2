#!/bin/bash

# Run LSTM training with multiple prediction lengths (comparable with TimeMixer)
echo "=================================="
echo "LSTM Training for ETTh1 (Comparable with TimeMixer)"
echo "=================================="
echo ""
echo "This script will train LSTM models with prediction lengths:"
echo "- 96, 192, 336, 720 (matching TimeMixer experiments)"
echo ""
echo "Dataset: ETTh1.csv"
echo "Sequence Length: 96"
echo "Features: 7 (multivariate)"
echo ""
echo "=================================="
echo ""

# Check if required files exist
if [ ! -f "dataset/ETT-small/ETTh1.csv" ]; then
    echo "Error: ETTh1.csv not found"
    exit 1
fi

if [ ! -f "config_etth1.json" ]; then
    echo "Error: config_etth1.json not found"
    exit 1
fi

# Run the comparable training script
python run_lstm_etth1_comparable.py

echo ""
echo "=================================="
echo "Training completed!"
echo "Results saved to: results/lstm_etth1/"
echo "=================================="
