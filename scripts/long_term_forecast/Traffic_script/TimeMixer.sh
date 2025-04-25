export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer
root_path=./dataset/traffic/
data_path=traffic.csv
data=custom
features=M
seq_len=96
label_len=0
e_layers=3
d_layers=1
factor=3
enc_in=862
dec_in=862
c_out=862
des='Exp'
itr=1
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8
down_sampling_method=avg

log_file="long_term_forecast_timeMixer_traffic_results.log"
echo "Training Results" > $log_file

for pred_len in 96 192 336 720; do
    model_id="traffic_${seq_len}_${pred_len}"

    echo "Running training for prediction length: $pred_len" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model $model_name \
      --data $data \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor $factor \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --des $des \
      --itr $itr \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method $down_sampling_method \
      --down_sampling_window $down_sampling_window | tee -a $log_file

      echo "Completed training for pred_len=$pred_len" | tee -a $log_file

done