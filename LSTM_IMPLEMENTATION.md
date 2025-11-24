# LSTM Model untuk Time Series Forecasting ETTh1

## Ringkasan Implementasi

Model LSTM telah berhasil diintegrasikan ke dalam Time-Series-Library untuk membandingkan performanya dengan TimeMixer.

## File yang Dibuat

### 1. Model LSTM (`/models/LSTM.py`)
- **Arsitektur**: Encoder-Decoder LSTM
- **Karakteristik**:
  - LSTM Encoder: Memproses input sequence dan menghasilkan hidden state
  - LSTM Decoder: Menggunakan hidden state dari encoder untuk generate predictions
  - DataEmbedding: Menggunakan temporal encoding yang sama dengan model lain
  - Projection Layer: Linear layer untuk output final

### 2. Training Script (`/scripts/long_term_forecast/ETT_script/LSTM_ETTh1.sh`)
- Training untuk 4 prediction horizons: 96, 192, 336, 720
- Konfigurasi yang comparable dengan TimeMixer

### 3. Integrasi ke Framework (`/exp/exp_basic.py`)
- LSTM ditambahkan ke model_dict
- Import statement ditambahkan

## Konfigurasi Model

### Hyperparameters LSTM:
```bash
seq_len=96              # Input sequence length
pred_len=[96,192,336,720]  # Prediction horizons
d_model=64              # Hidden size (lebih besar dari TimeMixer d_model=16)
e_layers=2              # Number of LSTM layers
dropout=0.1             # Dropout rate
learning_rate=0.001     # Learning rate
batch_size=128          # Batch size (sama dengan TimeMixer)
train_epochs=10         # Number of epochs
patience=10             # Early stopping patience
```

### Konfigurasi TimeMixer (untuk perbandingan):
```bash
seq_len=96
pred_len=[96,192,336,720]
d_model=16
e_layers=2
down_sampling_layers=3
learning_rate=0.01
batch_size=128
train_epochs=10
patience=10
```

## Cara Menjalankan

### 1. Pastikan Environment Sudah Disetup
```bash
cd /mnt/extended-home/galih/Time-Series-Library
pip install -r requirements.txt
```

### 2. Jalankan Training LSTM
```bash
bash scripts/long_term_forecast/ETT_script/LSTM_ETTh1.sh
```

### 3. Hasil akan tersimpan di:
- **Checkpoints**: `./checkpoints/long_term_forecast_ETTh1_96_{pred_len}_LSTM_*/`
- **Log file**: `long_term_forecast_LSTM_ETTh1_results.log`

## Perbandingan dengan TimeMixer

### Hasil TimeMixer ETTh1 (dari result_long_term_forecast_TimeMixer.txt):
| Pred Length | MSE      | MAE      |
|-------------|----------|----------|
| 96          | 0.3826   | 0.3995   |
| 192         | 0.4412   | 0.4292   |
| 336         | 0.4985   | 0.4592   |
| 720         | 0.4790   | 0.4727   |

### Hasil LSTM ETTh1:
Akan tersedia setelah training selesai.

## Perbedaan Arsitektur

### TimeMixer:
- Multi-scale temporal mixing
- Downsampling dengan averaging
- Seasonal-trend decomposition
- Direct multi-horizon forecasting

### LSTM (Implementasi ini):
- Sequential processing dengan recurrent connections
- Encoder-decoder architecture
- Direct output untuk semua timesteps
- Simpler architecture, lebih sedikit parameter

## Expected Performance

LSTM sebagai baseline model biasanya:
- ✅ Lebih mudah di-train
- ✅ Lebih cepat inference
- ✅ Lebih sedikit parameter
- ⚠️ Mungkin sedikit lebih buruk dalam long-horizon (720)
- ⚠️ Kurang baik dalam menangkap multi-scale patterns

## Troubleshooting

Jika error saat running:
1. Cek apakah torch terinstall: `pip show torch`
2. Cek CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Adjust batch_size jika OOM error

## Next Steps

1. ✅ Model LSTM sudah terintegrasi
2. ⏳ Jalankan training LSTM
3. ⏳ Bandingkan hasil MSE dan MAE dengan TimeMixer
4. ⏳ Analisis kelebihan/kekurangan masing-masing model
