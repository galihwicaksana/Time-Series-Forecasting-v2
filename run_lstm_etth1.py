"""
LSTM Time Series Prediction for ETTh1 Dataset
This script trains and evaluates an LSTM model on the ETTh1 dataset
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

# Add LSTM model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LSTM-Neural-Network-for-Time-Series-Prediction'))

from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data, save_path=None):
    """Plot prediction results"""
    fig = plt.figure(figsize=(12, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', alpha=0.8)
    ax.plot(predicted_data, label='Prediction', alpha=0.8)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title('LSTM Prediction vs True Data - ETTh1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'[Plot] Saved to {save_path}')
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len, save_path=None):
    """Plot multiple sequence predictions"""
    fig = plt.figure(figsize=(15, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', alpha=0.8)
    
    # Plot predictions with padding
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label=f'Prediction {i+1}' if i < 3 else '', alpha=0.6)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title('LSTM Multiple Sequence Predictions - ETTh1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'[Plot] Saved to {save_path}')
    plt.show()


def calculate_metrics(predictions, actuals):
    """Calculate prediction metrics"""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # MAPE (avoid division by zero)
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.any() else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def main():
    # Load configuration
    config_path = 'config_etth1.json'
    print(f'[Config] Loading configuration from {config_path}')
    configs = json.load(open(config_path, 'r'))
    
    # Create save directory
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
        print(f"[Setup] Created directory: {configs['model']['save_dir']}")
    
    # Load data
    print('[Data] Loading ETTh1 dataset...')
    data = DataLoader(
        configs['data']['filename'],
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    
    print(f'[Data] Train samples: {data.len_train}')
    print(f'[Data] Test samples: {data.len_test}')
    print(f'[Data] Features: {len(configs["data"]["columns"])}')
    print(f'[Data] Sequence length: {configs["data"]["sequence_length"]}')
    
    # Build model
    print('[Model] Building LSTM model...')
    model = Model()
    model.build_model(configs)
    
    # Prepare training data
    print('[Training] Preparing training data...')
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']
    )
    
    # Train model using generator (memory efficient)
    print('[Training] Starting training...')
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    
    # Prepare test data
    print('[Testing] Preparing test data...')
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    print(f'[Testing] Test data shape: x={x_test.shape}, y={y_test.shape}')
    
    # Make predictions
    print('[Prediction] Generating predictions...')
    
    # Point-by-point prediction
    predictions_point = model.predict_point_by_point(x_test)
    
    # Calculate metrics
    print('\n' + '='*50)
    print('PREDICTION RESULTS')
    print('='*50)
    
    metrics = calculate_metrics(predictions_point, y_test)
    print(f'MSE:  {metrics["MSE"]:.6f}')
    print(f'RMSE: {metrics["RMSE"]:.6f}')
    print(f'MAE:  {metrics["MAE"]:.6f}')
    print(f'MAPE: {metrics["MAPE"]:.2f}%')
    print('='*50 + '\n')
    
    # Save metrics
    metrics_path = os.path.join(configs['model']['save_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'[Metrics] Saved to {metrics_path}')
    
    # Plot results
    print('[Plotting] Generating plots...')
    plot_path = os.path.join(configs['model']['save_dir'], 'prediction_results.png')
    plot_results(predictions_point, y_test, save_path=plot_path)
    
    # Optional: Multiple sequence prediction
    try:
        print('[Prediction] Generating multiple sequence predictions...')
        predictions_multi = model.predict_sequences_multiple(
            x_test, 
            configs['data']['sequence_length'], 
            configs['data']['sequence_length']
        )
        
        plot_path_multi = os.path.join(configs['model']['save_dir'], 'prediction_multiple.png')
        plot_results_multiple(
            predictions_multi, 
            y_test, 
            configs['data']['sequence_length'],
            save_path=plot_path_multi
        )
    except Exception as e:
        print(f'[Warning] Multiple sequence prediction failed: {e}')
    
    print('\n[Complete] Training and evaluation finished successfully!')
    print(f'[Output] Results saved to: {configs["model"]["save_dir"]}')


if __name__ == '__main__':
    main()
