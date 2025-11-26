# LSTM Time Series Prediction for ETTh1 Dataset

This directory contains scripts and configuration for training an LSTM model on the ETTh1 (Electricity Transformer Temperature) dataset.

## Dataset Information

**ETTh1.csv** contains hourly data with the following features:
- `HUFL`: High UseFul Load
- `HULL`: High UseLess Load  
- `MUFL`: Middle UseFul Load
- `MULL`: Middle UseLess Load
- `LUFL`: Low UseFul Load
- `LULL`: Low UseLess Load
- `OT`: Oil Temperature (target variable)

**Total records**: 17,422 hourly observations

## Files Created

1. **config_etth1.json** - Configuration file with model and data parameters
2. **run_lstm_etth1.py** - Main training and evaluation script
3. **run_lstm_etth1.sh** - Shell script to execute training

## Configuration

### Data Parameters
- **Sequence length**: 96 timesteps (4 days of hourly data)
- **Train/Test split**: 70/30
- **Features**: All 7 columns (multivariate)
- **Normalization**: Enabled

### Model Architecture
- **Layer 1**: LSTM (128 neurons) + Dropout (0.2)
- **Layer 2**: LSTM (64 neurons) + Dropout (0.2)
- **Layer 3**: LSTM (32 neurons) + Dropout (0.2)
- **Output**: Dense layer (1 neuron, linear activation)

### Training Parameters
- **Epochs**: 50
- **Batch size**: 32
- **Loss function**: MSE (Mean Squared Error)
- **Optimizer**: Adam

## How to Run

### Option 1: Using Shell Script
```bash
./run_lstm_etth1.sh
```

### Option 2: Direct Python Execution
```bash
python run_lstm_etth1.py
```

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install numpy pandas matplotlib keras tensorflow
```

Or use the requirements file:
```bash
pip install -r LSTM-Neural-Network-for-Time-Series-Prediction/requirements.txt
```

## Output

The script will generate:

1. **Trained model**: Saved in `checkpoints/lstm_etth1/`
2. **Metrics JSON**: Performance metrics (MSE, RMSE, MAE, MAPE)
3. **Plots**:
   - `prediction_results.png` - Point-by-point predictions
   - `prediction_multiple.png` - Multiple sequence predictions

## Evaluation Metrics

The model outputs the following metrics:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Customization

You can modify the configuration in `config_etth1.json`:

- Change `sequence_length` for different lookback windows
- Adjust `train_test_split` ratio
- Modify model architecture in the `layers` section
- Change training parameters (`epochs`, `batch_size`)

## Example Output

```
[Data] Train samples: 12195
[Data] Test samples: 5227
[Data] Features: 7
[Data] Sequence length: 96

[Training] Starting training...
[Model] Training Started
[Model] 50 epochs, 32 batch size, 381 batches per epoch
...

PREDICTION RESULTS
==================================================
MSE:  0.123456
RMSE: 0.351364
MAE:  0.287432
MAPE: 5.43%
==================================================
```

## Notes

- Training time depends on your hardware (CPU/GPU)
- The model uses generator-based training for memory efficiency
- Early stopping is implemented in the base model
- All predictions are on normalized data space

## Troubleshooting

If you encounter issues:

1. Check that the dataset path is correct in `config_etth1.json`
2. Ensure all dependencies are installed
3. Verify that the LSTM model directory exists
4. Check that you have write permissions for the checkpoint directory
