"""
Compare LSTM and TimeMixer Results on ETTh1 Dataset
This script loads and compares the prediction results from both models
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_lstm_results():
    """Load LSTM results from summary file"""
    lstm_path = 'results/lstm_etth1/summary_results.json'
    
    if not os.path.exists(lstm_path):
        print(f"[Warning] LSTM results not found at {lstm_path}")
        return None
    
    with open(lstm_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for item in data['results']:
        results[item['pred_len']] = {
            'mse': item['mse'],
            'mae': item['mae']
        }
    
    return results


def load_timemixer_results():
    """Load TimeMixer results from checkpoint directories"""
    results = {}
    pred_lens = [96, 192, 336, 720]
    
    for pred_len in pred_lens:
        checkpoint_dir = f'checkpoints/long_term_forecast_ETTh1_96_{pred_len}_TimeMixer_custom_ftM_sl96_ll0_pl{pred_len}_dm16_nh8_el3_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
        
        result_file = os.path.join(checkpoint_dir, 'result.txt')
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                content = f.read()
                # Parse metrics (format: "mse:0.123, mae:0.456")
                try:
                    parts = content.split(',')
                    mse = float(parts[0].split(':')[1].strip())
                    mae = float(parts[1].split(':')[1].strip())
                    results[pred_len] = {'mse': mse, 'mae': mae}
                except:
                    print(f"[Warning] Could not parse TimeMixer results for pred_len={pred_len}")
        else:
            print(f"[Warning] TimeMixer results not found for pred_len={pred_len}")
    
    return results if results else None


def create_comparison_table(lstm_results, timemixer_results):
    """Create comparison table"""
    
    print('\n' + '='*80)
    print('MODEL COMPARISON: LSTM vs TimeMixer on ETTh1')
    print('='*80)
    print(f'{"Pred Len":<12} {"LSTM MSE":<15} {"TimeMixer MSE":<15} {"LSTM MAE":<15} {"TimeMixer MAE":<15}')
    print('-'*80)
    
    comparison_data = []
    
    for pred_len in sorted(lstm_results.keys()):
        lstm_mse = lstm_results[pred_len]['mse']
        lstm_mae = lstm_results[pred_len]['mae']
        
        if timemixer_results and pred_len in timemixer_results:
            tm_mse = timemixer_results[pred_len]['mse']
            tm_mae = timemixer_results[pred_len]['mae']
            
            # Calculate improvement
            mse_improvement = ((lstm_mse - tm_mse) / tm_mse) * 100
            mae_improvement = ((lstm_mae - tm_mae) / tm_mae) * 100
            
            print(f'{pred_len:<12} {lstm_mse:<15.6f} {tm_mse:<15.6f} {lstm_mae:<15.6f} {tm_mae:<15.6f}')
            
            comparison_data.append({
                'pred_len': pred_len,
                'lstm_mse': lstm_mse,
                'timemixer_mse': tm_mse,
                'lstm_mae': lstm_mae,
                'timemixer_mae': tm_mae,
                'mse_improvement_%': mse_improvement,
                'mae_improvement_%': mae_improvement
            })
        else:
            print(f'{pred_len:<12} {lstm_mse:<15.6f} {"N/A":<15} {lstm_mae:<15.6f} {"N/A":<15}')
            
            comparison_data.append({
                'pred_len': pred_len,
                'lstm_mse': lstm_mse,
                'timemixer_mse': None,
                'lstm_mae': lstm_mae,
                'timemixer_mae': None,
                'mse_improvement_%': None,
                'mae_improvement_%': None
            })
    
    print('='*80)
    
    if timemixer_results:
        print('\nImprovement: Negative % = LSTM is better, Positive % = TimeMixer is better')
        print('-'*80)
        for data in comparison_data:
            if data['mse_improvement_%'] is not None:
                mse_imp = data['mse_improvement_%']
                mae_imp = data['mae_improvement_%']
                better_mse = "LSTM" if mse_imp < 0 else "TimeMixer"
                better_mae = "LSTM" if mae_imp < 0 else "TimeMixer"
                
                print(f'Pred Len {data["pred_len"]}: MSE {abs(mse_imp):.2f}% better ({better_mse}), '
                      f'MAE {abs(mae_imp):.2f}% better ({better_mae})')
        print('='*80)
    
    return comparison_data


def plot_comparison(comparison_data, save_path='results/lstm_etth1/comparison_plot.png'):
    """Create comparison plots"""
    
    if not comparison_data:
        print("[Warning] No comparison data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    pred_lens = [d['pred_len'] for d in comparison_data]
    lstm_mse = [d['lstm_mse'] for d in comparison_data]
    lstm_mae = [d['lstm_mae'] for d in comparison_data]
    
    # Check if TimeMixer data exists
    has_timemixer = any(d['timemixer_mse'] is not None for d in comparison_data)
    
    if has_timemixer:
        tm_mse = [d['timemixer_mse'] if d['timemixer_mse'] is not None else 0 for d in comparison_data]
        tm_mae = [d['timemixer_mae'] if d['timemixer_mae'] is not None else 0 for d in comparison_data]
        
        # MSE comparison
        x = np.arange(len(pred_lens))
        width = 0.35
        
        ax1.bar(x - width/2, lstm_mse, width, label='LSTM', alpha=0.8)
        ax1.bar(x + width/2, tm_mse, width, label='TimeMixer', alpha=0.8)
        ax1.set_xlabel('Prediction Length')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE Comparison: LSTM vs TimeMixer')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pred_lens)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE comparison
        ax2.bar(x - width/2, lstm_mae, width, label='LSTM', alpha=0.8)
        ax2.bar(x + width/2, tm_mae, width, label='TimeMixer', alpha=0.8)
        ax2.set_xlabel('Prediction Length')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE Comparison: LSTM vs TimeMixer')
        ax2.set_xticks(x)
        ax2.set_xticklabels(pred_lens)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Only LSTM data
        ax1.plot(pred_lens, lstm_mse, marker='o', linewidth=2, markersize=8, label='LSTM')
        ax1.set_xlabel('Prediction Length')
        ax1.set_ylabel('MSE')
        ax1.set_title('LSTM MSE by Prediction Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(pred_lens, lstm_mae, marker='o', linewidth=2, markersize=8, label='LSTM', color='orange')
        ax2.set_xlabel('Prediction Length')
        ax2.set_ylabel('MAE')
        ax2.set_title('LSTM MAE by Prediction Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n[Plot] Comparison plot saved to {save_path}')
    plt.close()


def main():
    print('\n' + '#'*80)
    print('COMPARING LSTM AND TIMEMIXER RESULTS')
    print('#'*80 + '\n')
    
    # Load results
    print('[Loading] LSTM results...')
    lstm_results = load_lstm_results()
    
    print('[Loading] TimeMixer results...')
    timemixer_results = load_timemixer_results()
    
    if lstm_results is None:
        print('\n[Error] No LSTM results found!')
        print('[Info] Please run: python run_lstm_etth1_comparable.py first')
        return
    
    # Create comparison
    comparison_data = create_comparison_table(lstm_results, timemixer_results)
    
    # Save comparison to CSV
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        csv_path = 'results/lstm_etth1/comparison_results.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f'\n[Results] Comparison saved to {csv_path}')
    
    # Create plots
    plot_comparison(comparison_data)
    
    print('\n' + '#'*80)
    print('COMPARISON COMPLETED!')
    print('#'*80 + '\n')
    
    if timemixer_results is None:
        print('[Note] TimeMixer results not found. Run TimeMixer experiments to enable full comparison.')
        print('[Command] bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh')


if __name__ == '__main__':
    main()
