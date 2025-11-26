"""
LSTM Time Series Prediction for ETTh1 Dataset - Comparable with TimeMixer
This script trains and evaluates LSTM with multiple prediction lengths to match TimeMixer
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add LSTM model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LSTM-Neural-Network-for-Time-Series-Prediction'))

# Suppress TensorFlow warnings
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from core.data_processor import DataLoader
from core.model import Model


def calculate_metrics(predictions, actuals):
    """Calculate prediction metrics matching TimeMixer format"""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    
    return {
        'MSE': float(mse),
        'MAE': float(mae)
    }


def plot_comparison(predictions, actuals, pred_len, save_path):
    """Plot predictions vs actuals"""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot only first 500 points for clarity
    max_points = min(500, len(actuals))
    
    ax.plot(actuals[:max_points], label='Ground Truth', alpha=0.8, linewidth=1.5)
    ax.plot(predictions[:max_points], label='LSTM Prediction', alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value (Normalized)')
    ax.set_title(f'LSTM Prediction (pred_len={pred_len}) - ETTh1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved to {save_path}')


def train_and_evaluate_model(seq_len, pred_len, configs):
    """Train and evaluate model for specific prediction length"""
    
    print(f'\n{"="*70}')
    print(f'Training LSTM for seq_len={seq_len}, pred_len={pred_len}')
    print(f'{"="*70}')
    
    # Update config for this run
    configs['data']['sequence_length'] = seq_len
    configs['data']['pred_len'] = pred_len
    
    # Adjust input_timesteps (sequence_length - 1)
    for layer in configs['model']['layers']:
        if layer['type'] == 'lstm' and 'input_timesteps' in layer:
            layer['input_timesteps'] = seq_len - 1
            break
    
    # Update save directory
    model_id = f'ETTh1_{seq_len}_{pred_len}'
    configs['model']['save_dir'] = f'checkpoints/lstm_etth1_{seq_len}_{pred_len}'
    
    # Create save directory
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
    
    # Load data
    print('[Data] Loading ETTh1 dataset...')
    data = DataLoader(
        configs['data']['filename'],
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    
    print(f'[Data] Train samples: {data.len_train}')
    print(f'[Data] Test samples: {data.len_test}')
    
    # Build model
    print('[Model] Building LSTM model...')
    model = Model()
    model.build_model(configs)
    
    # Train model
    print('[Training] Starting training...')
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']
    )
    
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
    
    # Make predictions
    print('[Prediction] Generating predictions...')
    
    if pred_len == 1:
        # Single-step prediction
        predictions = model.predict_point_by_point(x_test)
        actuals = y_test.flatten()
    else:
        # Multi-step prediction using sequence prediction
        print(f'[Prediction] Multi-step prediction for {pred_len} steps...')
        predictions_multi = model.predict_sequences_multiple(
            x_test, 
            configs['data']['sequence_length'], 
            pred_len
        )
        
        # Flatten predictions
        predictions = np.array(predictions_multi).flatten()
        
        # Adjust actuals to match prediction length
        actuals = []
        for i in range(len(predictions_multi)):
            start_idx = i * pred_len
            end_idx = start_idx + pred_len
            if end_idx <= len(y_test):
                actuals.extend(y_test[start_idx:end_idx].flatten())
        actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actuals)
    
    print(f'\n{"="*50}')
    print(f'RESULTS for pred_len={pred_len}')
    print(f'{"="*50}')
    print(f'MSE: {metrics["MSE"]:.6f}')
    print(f'MAE: {metrics["MAE"]:.6f}')
    print(f'{"="*50}\n')
    
    # Save metrics
    metrics_path = os.path.join(configs['model']['save_dir'], 'metrics.json')
    metrics['model_id'] = model_id
    metrics['seq_len'] = seq_len
    metrics['pred_len'] = pred_len
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'[Metrics] Saved to {metrics_path}')
    
    # Plot results
    plot_path = os.path.join(configs['model']['save_dir'], f'prediction_{model_id}.png')
    plot_comparison(predictions, actuals, pred_len, plot_path)
    
    return metrics


def main():
    # Load base configuration
    config_path = 'config_etth1.json'
    print(f'[Config] Loading configuration from {config_path}')
    
    with open(config_path, 'r') as f:
        base_configs = json.load(f)
    
    # Set training parameters to match TimeMixer experiment
    base_configs['training']['epochs'] = 10  # Match TimeMixer
    base_configs['training']['batch_size'] = 32
    
    # Prediction lengths to match TimeMixer
    seq_len = 96
    pred_lens = [96, 192, 336, 720]
    
    print(f'\n{"#"*70}')
    print('LSTM Training for Multiple Prediction Lengths (Comparable with TimeMixer)')
    print(f'{"#"*70}')
    print(f'Sequence Length: {seq_len}')
    print(f'Prediction Lengths: {pred_lens}')
    print(f'Dataset: ETTh1')
    print(f'Features: 7 (multivariate)')
    print(f'{"#"*70}\n')
    
    # Store all results
    all_results = []
    
    # Train and evaluate for each prediction length
    for pred_len in pred_lens:
        try:
            metrics = train_and_evaluate_model(seq_len, pred_len, base_configs.copy())
            all_results.append({
                'pred_len': pred_len,
                'mse': metrics['MSE'],
                'mae': metrics['MAE']
            })
        except Exception as e:
            print(f'[Error] Failed for pred_len={pred_len}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary results
    summary_dir = 'results/lstm_etth1'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    summary_path = os.path.join(summary_dir, 'summary_results.json')
    summary_data = {
        'model': 'LSTM',
        'dataset': 'ETTh1',
        'seq_len': seq_len,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    # Create comparison table
    print(f'\n{"="*70}')
    print('SUMMARY RESULTS - LSTM on ETTh1')
    print(f'{"="*70}')
    print(f'{"Pred Length":<15} {"MSE":<15} {"MAE":<15}')
    print(f'{"-"*70}')
    for result in all_results:
        print(f'{result["pred_len"]:<15} {result["mse"]:<15.6f} {result["mae"]:<15.6f}')
    print(f'{"="*70}\n')
    
    # Save as CSV for easy comparison
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(summary_dir, 'lstm_etth1_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'[Results] Saved to {csv_path}')
    print(f'[Results] Summary saved to {summary_path}')
    
    print(f'\n{"#"*70}')
    print('ALL TRAINING COMPLETED!')
    print('You can now compare these results with TimeMixer outputs')
    print(f'{"#"*70}\n')


if __name__ == '__main__':
    main()
