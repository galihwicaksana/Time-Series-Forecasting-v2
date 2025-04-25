export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
root_path=./dataset/traffic/
data_path=traffic.csv
data=custom
features=M
seq_len=96
label_len=48
e_layers=2
d_layers=1
factor=3
enc_in=862
dec_in=862
c_out=862
d_model=512
d_ff=512
top_k=5
des='Exp'
itr=1

log_file="long_term_forecast_timesNet_traffic_results.log"
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
        --d_model $d_model \
        --d_ff $d_ff \
        --top_k $top_k \
        --des $des \
        --itr $itr | tee -a $log_file

      echo "Completed training for pred_len=$pred_len" | tee -a $log_file

done