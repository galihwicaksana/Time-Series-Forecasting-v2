# LSTM Training - Status dan Hasil

## 🚀 Training Sedang Berjalan!

Training LSTM untuk dataset ETTh1 sedang berjalan di tmux session dengan konfigurasi:

### Konfigurasi Training
- **Dataset**: ETTh1.csv (12,194 train samples, 5,226 test samples)
- **Features**: 7 variabel (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Sequence Length**: 96 timesteps
- **Prediction Lengths**: 96, 192, 336, 720
- **Model Architecture**: 3-layer LSTM (128→64→32 neurons)
- **Epochs per model**: 10
- **Batch Size**: 32

### 📁 File dan Direktori

**Log Files:**
- `logs/lstm_etth1/training_YYYYMMDD_HHMMSS.log` - Training logs

**Model Checkpoints:**
- `checkpoints/lstm_etth1_96_96/` - Model untuk pred_len=96
- `checkpoints/lstm_etth1_96_192/` - Model untuk pred_len=192
- `checkpoints/lstm_etth1_96_336/` - Model untuk pred_len=336
- `checkpoints/lstm_etth1_96_720/` - Model untuk pred_len=720

**Results:**
- `results/lstm_etth1/summary_results.json` - Summary semua hasil
- `results/lstm_etth1/lstm_etth1_results.csv` - Hasil dalam format CSV
- `results/lstm_etth1/comparison_results.csv` - Perbandingan dengan TimeMixer (jika ada)

### 🛠️ Script yang Tersedia

**1. Monitor Training**
```bash
./monitor_training.sh
```
Menampilkan progress training real-time.

**2. Check Status**
```bash
./check_training_status.sh
```
Cek status lengkap: tmux session, hasil, models yang tersimpan.

**3. Attach ke Tmux Session**
```bash
tmux attach -t lstm_training
```
Masuk ke session untuk melihat training langsung.
- Tekan `Ctrl+b` lalu `d` untuk detach tanpa menghentikan training

**4. View Logs Real-time**
```bash
tail -f logs/lstm_etth1/training_LATEST.log
```
Lihat log secara real-time.

**5. Kill Training (jika perlu)**
```bash
tmux kill-session -t lstm_training
```

### 📊 Setelah Training Selesai

**1. Lihat Hasil**
```bash
./check_training_status.sh
```

**2. Bandingkan dengan TimeMixer**
```bash
python compare_lstm_timemixer.py
```

Ini akan:
- Load hasil LSTM dari `results/lstm_etth1/`
- Load hasil TimeMixer dari checkpoints
- Membuat tabel perbandingan MSE & MAE
- Generate comparison plot
- Save ke `results/lstm_etth1/comparison_results.csv`

### 📈 Format Output

**Summary Results JSON:**
```json
{
  "model": "LSTM",
  "dataset": "ETTh1",
  "seq_len": 96,
  "timestamp": "2025-11-24 14:51:00",
  "results": [
    {"pred_len": 96, "mse": 0.123, "mae": 0.234},
    {"pred_len": 192, "mse": 0.234, "mae": 0.345},
    {"pred_len": 336, "mse": 0.345, "mae": 0.456},
    {"pred_len": 720, "mse": 0.456, "mae": 0.567}
  ]
}
```

**CSV Results:**
```
pred_len,mse,mae
96,0.123456,0.234567
192,0.234567,0.345678
336,0.345678,0.456789
720,0.456789,0.567890
```

### ⏱️ Estimasi Waktu

- **Per epoch**: ~2-3 menit (tergantung hardware)
- **Per model**: ~20-30 menit (10 epochs)
- **Total untuk 4 models**: ~1.5-2 jam

### 🔧 Troubleshooting

**Jika training berhenti:**
1. Check logs: `tail -100 logs/lstm_etth1/training_*.log`
2. Check tmux: `tmux ls`
3. Restart training: `./run_lstm_training_tmux.sh`

**Jika memory error:**
- Turunkan batch_size di `config_etth1.json`
- Restart training

**Jika CUDA error:**
- Training akan otomatis fallback ke CPU
- Akan lebih lambat tapi tetap berjalan

### 📝 Notes

- Training berjalan di background dalam tmux
- Anda bisa logout dari SSH, training tetap berjalan
- Semua output tersimpan di log file
- Model checkpoint disimpan setiap selesai training
- Untuk re-run, hapus hasil sebelumnya atau ubah nama

### 🎯 Next Steps

Setelah training LSTM selesai:
1. Check hasil dengan `./check_training_status.sh`
2. Jalankan TimeMixer (jika belum): `bash scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh`
3. Bandingkan hasil: `python compare_lstm_timemixer.py`
4. Analisis hasil perbandingan
5. Dokumentasi untuk paper/thesis

---

**Log File Saat Ini**: `logs/lstm_etth1/training_20251124_145100.log`  
**Status**: Training RUNNING in tmux  
**Started**: 2025-11-24 14:51:00
