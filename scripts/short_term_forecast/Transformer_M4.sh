export CUDA_VISIBLE_DEVICES=0

model_name=Transformer
root_path=./dataset/m4
data=m4
features=M
e_layers=2
d_layers=1
factor=3
enc_in=1
dec_in=1
c_out=1
batch_size=16
d_model=512
des='Exp'
itr=1
learning_rate=0.001
loss='SMAPE'

log_file="short_term_forecast_transformer_M4_results.log"
echo "Training Results" > $log_file

seasonal_patterns=("Monthly" "Yearly" "Quarterly" "Weekly" "Daily" "Hourly")

for pattern in "${seasonal_patterns[@]}"; do
  model_id="m4_${pattern}"

  echo "Running training for seasonal pattern: $pattern" | tee -a $log_file

  python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --seasonal_patterns $pattern \
      --model_id $model_id \
      --model $model_name \
      --data $data \
      --features $features \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --factor $factor \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --batch_size $batch_size \
      --d_model $d_model \
      --des $des \
      --itr $itr \
      --learning_rate $learning_rate \
      --loss $loss | tee -a $log_file

      echo "Completed training for seasonal pattern=$pattern" | tee -a $log_file

done