# 📊 Perbandingan LSTM vs TimeMixer - ETTh1 Dataset

Dokumentasi untuk membandingkan performa model LSTM dengan TimeMixer pada dataset ETTh1.

## 🔧 Konfigurasi yang Digunakan

### **File: config_etth1.json**

#### Data Configuration
```json
{
  "filename": "dataset/ETT-small/ETTh1.csv",
  "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
  "sequence_length": 96,
  "train_test_split": 0.7,
  "normalise": true
}
```

- **Dataset**: ETTh1.csv (17,422 samples)
- **Features**: 7 variabel (multivariate)
- **Input Sequence**: 96 timesteps
- **Training Split**: 70% train, 30% test
- **Normalisasi**: Enabled (window-based)

#### Model Configuration
```json
{
  "loss": "mse",
  "optimizer": "adam",
  "layers": [
    {"type": "lstm", "neurons": 128, "return_seq": true},
    {"type": "dropout", "rate": 0.2},
    {"type": "lstm", "neurons": 64, "return_seq": true},
    {"type": "dropout", "rate": 0.2},
    {"type": "lstm", "neurons": 32, "return_seq": false},
    {"type": "dropout", "rate": 0.2},
    {"type": "dense", "neurons": 1, "activation": "linear"}
  ]
}
```

- **Arsitektur**: 3-layer LSTM dengan dropout
- **Hidden Units**: 128 → 64 → 32
- **Dropout Rate**: 0.2 (20%)
- **Loss Function**: MSE
- **Optimizer**: Adam

#### Training Configuration
```json
{
  "epochs": 10,
  "batch_size": 32
}
```

- **Epochs**: 10 (disesuaikan dengan TimeMixer)
- **Batch Size**: 32
- **Training Method**: Generator-based (memory efficient)

---

## 📈 Prediction Lengths untuk Perbandingan

Model dilatih dengan 4 prediction lengths yang sama dengan TimeMixer:

| Prediction Length | Deskripsi |
|------------------|-----------|
| **96** | 4 hari (short-term) |
| **192** | 8 hari (medium-term) |
| **336** | 14 hari (long-term) |
| **720** | 30 hari (very long-term) |

---

## 🎯 Perbandingan Konfigurasi: LSTM vs TimeMixer

| Parameter | LSTM | TimeMixer |
|-----------|------|-----------|
| **Model Type** | Recurrent (LSTM) | Transformer-based |
| **Sequence Length** | 96 | 96 |
| **Prediction Lengths** | 96, 192, 336, 720 | 96, 192, 336, 720 |
| **Features** | 7 (multivariate) | 7 (multivariate) |
| **Hidden Dimensions** | 128→64→32 | d_model=16, d_ff=32 |
| **Layers** | 3 LSTM layers | e_layers=2 |
| **Batch Size** | 32 | 128 |
| **Epochs** | 10 | 10 |
| **Loss Function** | MSE | MSE |
| **Optimizer** | Adam | Adam (lr=0.01) |
| **Dropout** | 0.2 | - |
| **Down Sampling** | - | 3 layers, window=2 |

---

## 🚀 Cara Menjalankan

### 1. Training LSTM dengan Multiple Prediction Lengths

```bash
# Option 1: Via shell script
./run_lstm_comparable.sh

# Option 2: Langsung Python
python run_lstm_etth1_comparable.py
```

Script ini akan:
- Melatih model LSTM untuk pred_len: 96, 192, 336, 720
- Menyimpan model di `checkpoints/lstm_etth1_{seq_len}_{pred_len}/`
- Generate metrics (MSE, MAE) untuk setiap pred_len
- Membuat visualisasi hasil prediksi
- Menyimpan summary di `results/lstm_etth1/`

### 2. Training TimeMixer (untuk perbandingan)

```bash
# Jalankan script TimeMixer
bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
```

### 3. Membandingkan Hasil

```bash
# Setelah kedua model selesai training
python compare_lstm_timemixer.py
```

Script ini akan:
- Load hasil LSTM dan TimeMixer
- Membuat tabel perbandingan
- Menghitung improvement percentage
- Generate comparison plots
- Save hasil ke CSV

---

## 📁 Output Structure

```
Time-Series-Library/
├── checkpoints/
│   ├── lstm_etth1_96_96/          # LSTM pred_len=96
│   │   ├── checkpoint.h5
│   │   ├── metrics.json
│   │   └── prediction_ETTh1_96_96.png
│   ├── lstm_etth1_96_192/         # LSTM pred_len=192
│   ├── lstm_etth1_96_336/         # LSTM pred_len=336
│   └── lstm_etth1_96_720/         # LSTM pred_len=720
│
└── results/
    └── lstm_etth1/
        ├── summary_results.json        # Summary semua pred_len
        ├── lstm_etth1_results.csv      # Hasil dalam CSV
        ├── comparison_results.csv       # Perbandingan LSTM vs TimeMixer
        └── comparison_plot.png          # Visualisasi perbandingan
```

---

## 📊 Format Output Metrics

### Individual Metrics (per prediction length)
```json
{
  "MSE": 0.123456,
  "MAE": 0.234567,
  "model_id": "ETTh1_96_96",
  "seq_len": 96,
  "pred_len": 96,
  "timestamp": "2025-11-24 10:30:00"
}
```

### Summary Results
```json
{
  "model": "LSTM",
  "dataset": "ETTh1",
  "seq_len": 96,
  "results": [
    {"pred_len": 96, "mse": 0.123, "mae": 0.234},
    {"pred_len": 192, "mse": 0.234, "mae": 0.345},
    {"pred_len": 336, "mse": 0.345, "mae": 0.456},
    {"pred_len": 720, "mse": 0.456, "mae": 0.567}
  ]
}
```

### Comparison Results
```csv
pred_len,lstm_mse,timemixer_mse,lstm_mae,timemixer_mae,mse_improvement_%,mae_improvement_%
96,0.123,0.110,-11.82,-10.50
192,0.234,0.220,-6.36,-5.20
336,0.345,0.330,-4.55,-4.10
720,0.456,0.440,-3.64,-3.20
```

---

## 📝 Interpretasi Hasil

### Metrik Evaluasi

**MSE (Mean Squared Error)**
- Mengukur rata-rata kuadrat error
- Lebih sensitif terhadap outlier
- Semakin kecil semakin baik

**MAE (Mean Absolute Error)**
- Mengukur rata-rata absolute error
- Lebih robust terhadap outlier
- Semakin kecil semakin baik

### Improvement Percentage
```
Improvement % = ((LSTM - TimeMixer) / TimeMixer) × 100

Negatif (-) = LSTM lebih baik
Positif (+) = TimeMixer lebih baik
```

---

## 🔍 Analisis yang Bisa Dilakukan

1. **Performance Comparison**
   - Model mana yang lebih akurat untuk setiap prediction length?
   - Apakah LSTM atau TimeMixer lebih baik untuk short-term vs long-term?

2. **Scalability Analysis**
   - Bagaimana performa menurun seiring bertambahnya prediction length?
   - Model mana yang lebih stabil untuk prediksi jangka panjang?

3. **Trade-offs**
   - **LSTM**: Lebih sederhana, cepat training, memori efisien
   - **TimeMixer**: Lebih complex, bisa capture long-range dependencies

4. **Computational Cost**
   - Training time comparison
   - Inference speed
   - Memory usage

---

## 🛠️ Customization

### Mengubah Prediction Lengths
Edit file `run_lstm_etth1_comparable.py`:
```python
pred_lens = [96, 192, 336, 720]  # Ubah sesuai kebutuhan
```

### Mengubah Model Architecture
Edit `config_etth1.json` bagian layers:
```json
{
  "type": "lstm",
  "neurons": 256,  // Ubah jumlah neurons
  "return_seq": true
}
```

### Mengubah Training Parameters
```json
{
  "epochs": 20,      // Tambah epochs
  "batch_size": 64   // Ubah batch size
}
```

---

## ⚠️ Notes & Tips

1. **Memory Management**
   - LSTM menggunakan generator-based training untuk efisiensi memori
   - Untuk dataset lebih besar, turunkan batch_size

2. **Training Time**
   - LSTM training ~10-30 menit per prediction length (tergantung hardware)
   - TimeMixer biasanya lebih cepat dengan GPU

3. **Reproducibility**
   - Set random seed untuk hasil yang reproducible
   - Dokumentasikan environment (Python version, library versions)

4. **Validation**
   - Gunakan validation split untuk tuning hyperparameter
   - Implement early stopping untuk mencegah overfitting

---

## 📚 References

**LSTM Implementation**
- Based on: [LSTM-Neural-Network-for-Time-Series-Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction)
- Keras/TensorFlow backend

**TimeMixer**
- Paper: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
- Original implementation in Time-Series-Library

**ETTh1 Dataset**
- Electricity Transformer Temperature (hourly)
- Source: [Autoformer paper](https://arxiv.org/abs/2106.13008)

---

## 🎓 Expected Outcomes

Setelah menjalankan semua script, Anda akan mendapatkan:

✅ Trained LSTM models untuk 4 prediction lengths  
✅ Performance metrics (MSE, MAE) untuk setiap model  
✅ Comparison table LSTM vs TimeMixer  
✅ Visualization plots  
✅ CSV files untuk analisis lebih lanjut  
✅ Model checkpoints yang bisa di-reload  

**Gunakan hasil ini untuk:**
- Paper/thesis comparison
- Model selection untuk production
- Baseline untuk model development
- Understanding trade-offs antara model architectures

---

## 🆘 Troubleshooting

**Error: Module not found**
```bash
pip install numpy pandas matplotlib keras tensorflow
```

**Error: CUDA out of memory**
- Turunkan batch_size di config
- Gunakan CPU untuk training kecil

**Error: Dataset not found**
- Pastikan path di config benar
- Check ETTh1.csv ada di `dataset/ETT-small/`

**Prediction shape mismatch**
- Pastikan sequence_length konsisten
- Check input_timesteps = sequence_length - 1

---

**Created**: 2025-11-24  
**Last Updated**: 2025-11-24  
**Version**: 1.0
