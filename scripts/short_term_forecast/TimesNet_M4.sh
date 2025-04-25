export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

log_file="short_term_forecast_timesNet_M4_results.log"
echo "Training Results" > $log_file

seasonal_patterns=("Monthly" "Yearly" "Quarterly" "Daily" "Weekly" "Hourly")
model_ids=("m4_Monthly" "m4_Yearly" "m4_Quarterly" "m4_Daily" "m4_Weekly" "m4_Hourly")
d_models=(32 16 64 16 32 32)
d_ff=(32 32 64 16 32 32)

for i in "${!seasonal_patterns[@]}"; do
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns "${seasonal_patterns[$i]}" \
    --model_id "${model_ids[$i]}" \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model "${d_models[$i]}" \
    --d_ff "${d_ff[$i]}" \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' >> $log_file 2>&1
done