"""
Script untuk visualisasi hasil training dan perbandingan model
Dapat dijalankan setelah training selesai untuk menganalisis hasil
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style untuk visualisasi yang lebih bagus
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

class ResultsVisualizer:
    def __init__(self, dataset_name='ETTh1'):
        """
        Initialize visualizer
        Args:
            dataset_name: 'ETTh1' atau 'ETTh2'
        """
        self.dataset_name = dataset_name
        self.models = ['TimeMixer', 'TimesNet', 'Transformer']
        self.pred_lengths = [96, 192, 336, 720]
        self.metrics = ['MSE', 'MAE']
        self.results = {}
        
    def parse_log_file(self, log_file_path):
        """
        Parse log file untuk ekstrak metrics (MSE, MAE)
        """
        if not os.path.exists(log_file_path):
            print(f"Warning: Log file not found: {log_file_path}")
            return None
            
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        results = {}
        
        # Pattern untuk menangkap MSE dan MAE dari log
        # Contoh: "mse:0.389, mae:0.419"
        pattern = r'mse:(\d+\.?\d*),?\s*mae:(\d+\.?\d*)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if matches:
            # Ambil hasil terakhir (biasanya test results)
            for i, pred_len in enumerate(self.pred_lengths):
                if i < len(matches):
                    mse, mae = matches[i]
                    results[pred_len] = {
                        'MSE': float(mse),
                        'MAE': float(mae)
                    }
        
        return results if results else None
    
    def parse_results_folder(self, model_name):
        """
        Parse results dari folder results/ (jika ada file CSV)
        """
        results_dir = Path('results')
        if not results_dir.exists():
            return None
            
        results = {}
        for pred_len in self.pred_lengths:
            # Cari file hasil prediksi
            pattern = f"*{self.dataset_name}*{pred_len}*{model_name}*.csv"
            files = list(results_dir.glob(pattern))
            
            if files:
                # Baca file pertama yang cocok
                df = pd.read_csv(files[0])
                # Hitung metrics dari prediksi vs actual
                if 'pred' in df.columns and 'true' in df.columns:
                    mse = ((df['pred'] - df['true']) ** 2).mean()
                    mae = (df['pred'] - df['true']).abs().mean()
                    results[pred_len] = {'MSE': mse, 'MAE': mae}
        
        return results if results else None
    
    def load_results(self):
        """
        Load results dari log files atau results folder
        """
        log_files = {
            'TimeMixer': f'long_term_forecast_timeMixer_{self.dataset_name}_results.log',
            'TimesNet': f'long_term_forecast_timesNet_{self.dataset_name}_results.log',
            'Transformer': f'long_term_forecast_transformer_{self.dataset_name}_results.log'
        }
        
        for model in self.models:
            log_file = log_files.get(model)
            if log_file:
                # Coba parse dari log file
                results = self.parse_log_file(log_file)
                if results:
                    self.results[model] = results
                else:
                    # Coba parse dari results folder
                    results = self.parse_results_folder(model)
                    if results:
                        self.results[model] = results
                    else:
                        print(f"No results found for {model}")
        
        return len(self.results) > 0
    
    def plot_single_model(self, model_name, save_dir='visualizations'):
        """
        Visualisasi untuk satu model
        """
        if model_name not in self.results:
            print(f"No results available for {model_name}")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        results = self.results[model_name]
        pred_lens = sorted(results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} Performance on {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # Plot MSE
        mse_values = [results[pl]['MSE'] for pl in pred_lens]
        axes[0].plot(pred_lens, mse_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        axes[0].set_xlabel('Prediction Length', fontweight='bold')
        axes[0].set_ylabel('MSE', fontweight='bold')
        axes[0].set_title('Mean Squared Error', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(pred_lens)
        
        # Tambahkan nilai di atas titik
        for pl, mse in zip(pred_lens, mse_values):
            axes[0].text(pl, mse, f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot MAE
        mae_values = [results[pl]['MAE'] for pl in pred_lens]
        axes[1].plot(pred_lens, mae_values, marker='s', linewidth=2, markersize=8, color='#A23B72')
        axes[1].set_xlabel('Prediction Length', fontweight='bold')
        axes[1].set_ylabel('MAE', fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(pred_lens)
        
        # Tambahkan nilai di atas titik
        for pl, mae in zip(pred_lens, mae_values):
            axes[1].text(pl, mae, f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{model_name}_{self.dataset_name}_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_comparison(self, save_dir='visualizations'):
        """
        Visualisasi perbandingan 3 model
        """
        if len(self.results) == 0:
            print("No results available for comparison")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Line Plot Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Model Comparison on {self.dataset_name}', fontsize=16, fontweight='bold')
        
        colors = {'TimeMixer': '#FF6B6B', 'TimesNet': '#4ECDC4', 'Transformer': '#45B7D1'}
        markers = {'TimeMixer': 'o', 'TimesNet': 's', 'Transformer': '^'}
        
        # MSE Comparison
        for model in self.results:
            results = self.results[model]
            pred_lens = sorted(results.keys())
            mse_values = [results[pl]['MSE'] for pl in pred_lens]
            axes[0].plot(pred_lens, mse_values, marker=markers.get(model, 'o'), 
                        linewidth=2.5, markersize=8, label=model, color=colors.get(model))
        
        axes[0].set_xlabel('Prediction Length', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('MSE (Lower is Better)', fontweight='bold', fontsize=12)
        axes[0].set_title('Mean Squared Error Comparison', fontweight='bold', fontsize=13)
        axes[0].legend(loc='best', fontsize=11, framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(self.pred_lengths)
        
        # MAE Comparison
        for model in self.results:
            results = self.results[model]
            pred_lens = sorted(results.keys())
            mae_values = [results[pl]['MAE'] for pl in pred_lens]
            axes[1].plot(pred_lens, mae_values, marker=markers.get(model, 'o'), 
                        linewidth=2.5, markersize=8, label=model, color=colors.get(model))
        
        axes[1].set_xlabel('Prediction Length', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('MAE (Lower is Better)', fontweight='bold', fontsize=12)
        axes[1].set_title('Mean Absolute Error Comparison', fontweight='bold', fontsize=13)
        axes[1].legend(loc='best', fontsize=11, framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(self.pred_lengths)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_line_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        # 2. Bar Chart Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Detailed Model Comparison on {self.dataset_name}', fontsize=16, fontweight='bold')
        
        for idx, pred_len in enumerate(self.pred_lengths):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            models_list = []
            mse_list = []
            mae_list = []
            
            for model in self.results:
                if pred_len in self.results[model]:
                    models_list.append(model)
                    mse_list.append(self.results[model][pred_len]['MSE'])
                    mae_list.append(self.results[model][pred_len]['MAE'])
            
            x = np.arange(len(models_list))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, mse_list, width, label='MSE', color='#FF6B6B', alpha=0.8)
            bars2 = ax.bar(x + width/2, mae_list, width, label='MAE', color='#4ECDC4', alpha=0.8)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel('Error Value', fontweight='bold')
            ax.set_title(f'Prediction Length: {pred_len}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models_list)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Tambahkan nilai di atas bar
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_bar_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        # 3. Heatmap Comparison
        self._plot_heatmap_comparison(save_dir)
    
    def _plot_heatmap_comparison(self, save_dir):
        """
        Heatmap untuk perbandingan visual cepat
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f'Performance Heatmap - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # Buat dataframe untuk MSE dan MAE
        mse_data = []
        mae_data = []
        models_list = []
        
        for model in self.results:
            models_list.append(model)
            mse_row = []
            mae_row = []
            for pred_len in self.pred_lengths:
                if pred_len in self.results[model]:
                    mse_row.append(self.results[model][pred_len]['MSE'])
                    mae_row.append(self.results[model][pred_len]['MAE'])
                else:
                    mse_row.append(np.nan)
                    mae_row.append(np.nan)
            mse_data.append(mse_row)
            mae_data.append(mae_row)
        
        mse_df = pd.DataFrame(mse_data, index=models_list, columns=self.pred_lengths)
        mae_df = pd.DataFrame(mae_data, index=models_list, columns=self.pred_lengths)
        
        # MSE Heatmap
        sns.heatmap(mse_df, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[0], 
                   cbar_kws={'label': 'MSE'}, linewidths=0.5)
        axes[0].set_title('MSE Heatmap (Lower is Better)', fontweight='bold')
        axes[0].set_xlabel('Prediction Length', fontweight='bold')
        axes[0].set_ylabel('Model', fontweight='bold')
        
        # MAE Heatmap
        sns.heatmap(mae_df, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[1], 
                   cbar_kws={'label': 'MAE'}, linewidths=0.5)
        axes[1].set_title('MAE Heatmap (Lower is Better)', fontweight='bold')
        axes[1].set_xlabel('Prediction Length', fontweight='bold')
        axes[1].set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_heatmap_{self.dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def generate_summary_table(self, save_dir='visualizations'):
        """
        Generate summary table dalam format CSV dan console
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Buat summary dataframe
        summary_data = []
        for model in self.results:
            for pred_len in sorted(self.results[model].keys()):
                summary_data.append({
                    'Model': model,
                    'Prediction Length': pred_len,
                    'MSE': self.results[model][pred_len]['MSE'],
                    'MAE': self.results[model][pred_len]['MAE']
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, f'summary_{self.dataset_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary table: {csv_path}")
        
        # Print to console
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE - {self.dataset_name}")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Ranking
        print(f"\n{'='*80}")
        print(f"MODEL RANKING - {self.dataset_name}")
        print(f"{'='*80}")
        
        for pred_len in self.pred_lengths:
            print(f"\nPrediction Length: {pred_len}")
            pred_data = df[df['Prediction Length'] == pred_len].copy()
            
            if len(pred_data) > 0:
                print("\nBy MSE (Lower is Better):")
                pred_data_mse = pred_data.sort_values('MSE')
                for idx, row in enumerate(pred_data_mse.itertuples(), 1):
                    print(f"  {idx}. {row.Model}: {row.MSE:.4f}")
                
                print("\nBy MAE (Lower is Better):")
                pred_data_mae = pred_data.sort_values('MAE')
                for idx, row in enumerate(pred_data_mae.itertuples(), 1):
                    print(f"  {idx}. {row.Model}: {row.MAE:.4f}")
        
        print(f"\n{'='*80}\n")
        
        return df

def main():
    """
    Main function untuk menjalankan visualisasi
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training results and compare models')
    parser.add_argument('--dataset', type=str, default='ETTh1', choices=['ETTh1', 'ETTh2'],
                       help='Dataset name (ETTh1 or ETTh2)')
    parser.add_argument('--models', type=str, nargs='+', default=['TimeMixer', 'TimesNet', 'Transformer'],
                       help='Models to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Starting Visualization for {args.dataset}")
    print(f"{'='*80}\n")
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(dataset_name=args.dataset)
    
    # Load results
    print("Loading results...")
    if not visualizer.load_results():
        print("\n⚠️  No results found! Please make sure:")
        print("  1. Training has been completed")
        print("  2. Log files exist in the current directory")
        print("  3. Log files follow the naming convention:")
        print(f"     - long_term_forecast_timeMixer_{args.dataset}_results.log")
        print(f"     - long_term_forecast_timesNet_{args.dataset}_results.log")
        print(f"     - long_term_forecast_transformer_{args.dataset}_results.log")
        return
    
    print(f"✓ Loaded results for {len(visualizer.results)} model(s)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Individual model plots
    for model in visualizer.results:
        print(f"  - Creating visualization for {model}...")
        visualizer.plot_single_model(model, save_dir=args.output_dir)
    
    # Comparison plots
    if len(visualizer.results) > 1:
        print(f"  - Creating comparison visualizations...")
        visualizer.plot_comparison(save_dir=args.output_dir)
    
    # Summary table
    print("\nGenerating summary table...")
    visualizer.generate_summary_table(save_dir=args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations completed!")
    print(f"  Output directory: {args.output_dir}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
