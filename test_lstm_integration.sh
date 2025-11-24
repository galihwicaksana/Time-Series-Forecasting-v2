#!/bin/bash
# Quick test untuk memverifikasi LSTM model bisa dijalankan

cd /mnt/extended-home/galih/Time-Series-Library

echo "=== Testing LSTM Model Integration ==="
echo ""

# Test 1: Import model
echo "Test 1: Importing LSTM model..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from models.LSTM import Model
    print("✓ LSTM model imported successfully")
except Exception as e:
    print(f"✗ Error importing LSTM: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "Please install dependencies first:"
    echo "pip install torch numpy"
    exit 1
fi

# Test 2: Check integration with exp_basic
echo ""
echo "Test 2: Checking integration with exp_basic.py..."
grep -q "LSTM" exp/exp_basic.py && echo "✓ LSTM registered in model_dict" || echo "✗ LSTM not found in model_dict"

# Test 3: Check training script
echo ""
echo "Test 3: Checking training script..."
if [ -x scripts/long_term_forecast/ETT_script/LSTM_ETTh1.sh ]; then
    echo "✓ Training script is executable"
else
    echo "✗ Training script is not executable"
fi

# Test 4: Check data availability
echo ""
echo "Test 4: Checking ETTh1 dataset..."
if [ -f dataset/ETT-small/ETTh1.csv ]; then
    echo "✓ ETTh1.csv dataset found"
    wc -l dataset/ETT-small/ETTh1.csv
else
    echo "✗ ETTh1.csv not found in dataset/ETT-small/"
fi

echo ""
echo "=== Verification Complete ==="
echo ""
echo "To start training, run:"
echo "bash scripts/long_term_forecast/ETT_script/LSTM_ETTh1.sh"
