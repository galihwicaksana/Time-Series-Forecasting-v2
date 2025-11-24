export CUDA_VISIBLE_DEVICES=0

model_name=LSTM

seq_len=96
e_layers=2
learning_rate=0.001
d_model=64
d_ff=256
train_epochs=10
patience=10
batch_size=128
dropout=0.1

# Log file untuk hasil training
log_file="long_term_forecast_LSTM_ETTh1_results.log"
echo "Training Results" > $log_file

for pred_len in 96 192 336 720; do
    model_id="ETTh1_${seq_len}_${pred_len}"

    echo "Running training for prediction length: $pred_len" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --batch_size $batch_size \
      --dropout $dropout | tee -a $log_file

    echo "Completed training for pred_len=$pred_len" | tee -a $log_file

done
