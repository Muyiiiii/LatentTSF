#!/bin/bash
# Full sweep: reproduce paper Table 1 across (datasets × models × pred_lens).
# Re-uses the same per-dataset configs as run_train.sh — only the top
# "User-tunable" block (which datasets/models/horizons to run) differs.
#
# Outer-loop structure: dataset -> model -> pred_len.
# Skips runs whose AE checkpoint is missing (prints a warning).
# Each run appends its metrics to $result_csv / $result_txt.
#
# Hyperparameters follow the paper's main recipe:
#   - Loss weights: mse_weight=10 (α), cosine_weight=15 (β), perceptual=recon=0
#   - seq_len=720, lradj=cosine, AE pretrained with MAE loss (sl=24), frozen AE
#
# For a single quick example, use run_train.sh instead.

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # or "online" / "disabled"

# ============ User-tunable ============
# Space-separated lists. Defaults below sweep all 9 datasets, one model, all 4 horizons.
datasets="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity traffic Solar"
models="DLinear"
pred_lens="96 192 336 720"

result_csv="result.csv"
result_txt="result/result.txt"
delete_checkpoint=0  # 1: delete each run's checkpoint after testing (saves disk)
# ======================================

# Per-dataset config (paper main recipe). Keys: dataset name.
declare -A data_path        ; data_path["ETTh1"]="ETTh1.csv"
                              data_path["ETTh2"]="ETTh2.csv"
                              data_path["ETTm1"]="ETTm1.csv"
                              data_path["ETTm2"]="ETTm2.csv"
                              data_path["weather"]="weather.csv"
                              data_path["exchange_rate"]="exchange_rate.csv"
                              data_path["electricity"]="electricity.csv"
                              data_path["traffic"]="traffic.csv"
                              data_path["Solar"]="solar_AL.txt"

declare -A root_path        ; root_path["ETTh1"]="./dataset/ETT-small/"
                              root_path["ETTh2"]="./dataset/ETT-small/"
                              root_path["ETTm1"]="./dataset/ETT-small/"
                              root_path["ETTm2"]="./dataset/ETT-small/"
                              root_path["weather"]="./dataset/weather/"
                              root_path["exchange_rate"]="./dataset/exchange_rate/"
                              root_path["electricity"]="./dataset/electricity/"
                              root_path["traffic"]="./dataset/traffic/"
                              root_path["Solar"]="./dataset/Solar/"

declare -A data_name        ; data_name["ETTh1"]="ETTh1"
                              data_name["ETTh2"]="ETTh2"
                              data_name["ETTm1"]="ETTm1"
                              data_name["ETTm2"]="ETTm2"
                              data_name["weather"]="custom"
                              data_name["exchange_rate"]="custom"
                              data_name["electricity"]="custom"
                              data_name["traffic"]="custom"
                              data_name["Solar"]="Solar"

declare -A enc_in           ; enc_in["ETTh1"]=7    ; enc_in["ETTh2"]=7
                              enc_in["ETTm1"]=7    ; enc_in["ETTm2"]=7
                              enc_in["weather"]=21 ; enc_in["exchange_rate"]=8
                              enc_in["electricity"]=321 ; enc_in["traffic"]=862
                              enc_in["Solar"]=137

declare -A ae_d_model       ; ae_d_model["ETTh1"]=32   ; ae_d_model["ETTh2"]=64
                              ae_d_model["ETTm1"]=32   ; ae_d_model["ETTm2"]=64
                              ae_d_model["weather"]=64 ; ae_d_model["exchange_rate"]=128
                              ae_d_model["electricity"]=512 ; ae_d_model["traffic"]=512
                              ae_d_model["Solar"]=256

declare -A ae_d_ff          ; ae_d_ff["ETTh1"]=64   ; ae_d_ff["ETTh2"]=128
                              ae_d_ff["ETTm1"]=64   ; ae_d_ff["ETTm2"]=128
                              ae_d_ff["weather"]=128 ; ae_d_ff["exchange_rate"]=256
                              ae_d_ff["electricity"]=1024 ; ae_d_ff["traffic"]=1024
                              ae_d_ff["Solar"]=512

declare -A ae_des           ; ae_des["ETTh1"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["ETTh2"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["ETTm1"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["ETTm2"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["weather"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["exchange_rate"]="Exp-sl24-lr0.0005-500-32bs"
                              ae_des["electricity"]="Exp-lr0.0001-500-16bs"
                              ae_des["traffic"]="Exp-lr0.0001-500-16bs"
                              ae_des["Solar"]="Exp-sl24-lr0.0005-500-32bs"

declare -A batch_size       ; batch_size["ETTh1"]=256 ; batch_size["ETTh2"]=256
                              batch_size["ETTm1"]=256 ; batch_size["ETTm2"]=256
                              batch_size["weather"]=256 ; batch_size["exchange_rate"]=256
                              batch_size["electricity"]=32 ; batch_size["traffic"]=32
                              batch_size["Solar"]=32

declare -A learning_rate    ; learning_rate["ETTh1"]=0.0003 ; learning_rate["ETTh2"]=0.0003
                              learning_rate["ETTm1"]=0.0003 ; learning_rate["ETTm2"]=0.0003
                              learning_rate["weather"]=0.001 ; learning_rate["exchange_rate"]=0.001
                              learning_rate["electricity"]=0.001 ; learning_rate["traffic"]=0.0005
                              learning_rate["Solar"]=0.001

declare -A accum_steps      ; accum_steps["ETTh1"]=1 ; accum_steps["ETTh2"]=1
                              accum_steps["ETTm1"]=1 ; accum_steps["ETTm2"]=1
                              accum_steps["weather"]=1 ; accum_steps["exchange_rate"]=1
                              accum_steps["electricity"]=1 ; accum_steps["traffic"]=4
                              accum_steps["Solar"]=1

declare -A step_size        ; step_size["ETTh1"]=1 ; step_size["ETTh2"]=1
                              step_size["ETTm1"]=1 ; step_size["ETTm2"]=1
                              step_size["weather"]=1 ; step_size["exchange_rate"]=1
                              step_size["electricity"]=60 ; step_size["traffic"]=60
                              step_size["Solar"]=60

ae_seq_len=24
delete_flag=""
if [ "$delete_checkpoint" -eq 1 ]; then
  delete_flag="--delete_checkpoint"
fi

skipped=()
launched=0

for dataset in $datasets; do
  ae_path="./checkpoints/AutoEncoder_MLP_MAE_${dataset}_${data_name[$dataset]}_AE_${data_name[$dataset]}_ftM_sl${ae_seq_len}_dm${ae_d_model[$dataset]}_dff${ae_d_ff[$dataset]}_lradj0_${ae_des[$dataset]}_0/checkpoint.pth"

  if [ ! -f "$ae_path" ]; then
    echo "[skip] pretrained AE not found for $dataset: $ae_path"
    skipped+=("$dataset")
    continue
  fi

  for model in $models; do
    for pred_len in $pred_lens; do
      echo ""
      echo "================================================================"
      echo "[run] dataset=$dataset  model=$model  pred_len=$pred_len"
      echo "      batch=${batch_size[$dataset]}  lr=${learning_rate[$dataset]}  accum=${accum_steps[$dataset]}  step=${step_size[$dataset]}"
      echo "================================================================"

      python -u my_train.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "Latent_${model}_${dataset}_pl${pred_len}" \
        --model "$model" \
        --data "${data_name[$dataset]}" \
        --root_path "${root_path[$dataset]}" \
        --data_path "${data_path[$dataset]}" \
        --features M \
        --use_norm 1 \
        --seq_len 720 \
        --label_len 0 \
        --pred_len "$pred_len" \
        --step "${step_size[$dataset]}" \
        --enc_in "${enc_in[$dataset]}" \
        --dec_in "${enc_in[$dataset]}" \
        --c_out "${enc_in[$dataset]}" \
        --d_model "${ae_d_model[$dataset]}" \
        --d_ff "${ae_d_ff[$dataset]}" \
        --e_layers 2 \
        --d_layers 1 \
        --n_heads 4 \
        --dropout 0.1 \
        --train_epochs 100 \
        --batch_size "${batch_size[$dataset]}" \
        --accum_steps "${accum_steps[$dataset]}" \
        --learning_rate "${learning_rate[$dataset]}" \
        --patience 5 \
        --seed 42 \
        --use_lradj 1 \
        --lradj cosine \
        --use_latent \
        --encoder_type AE \
        --ae_type MLP \
        --ae_loss MAE \
        --autoencoder_path "$ae_path" \
        --load_pretrained_ae 1 \
        --freeze_encoder \
        --freeze_decoder \
        --mse_weight 10.0 \
        --cosine_weight 15.0 \
        --perceptual_weight 0.0 \
        --reconstruction_weight 0.0 \
        --des Exp \
        --itr 1 \
        --result_csv "$result_csv" \
        --result_txt "$result_txt" \
        $delete_flag

      launched=$((launched + 1))
    done
  done
done

echo ""
echo "================================================================"
echo "Sweep finished — launched $launched run(s)."
if [ ${#skipped[@]} -gt 0 ]; then
  echo "Skipped datasets (no AE checkpoint): ${skipped[*]}"
fi
echo "Results: $result_csv  /  $result_txt"
echo "================================================================"
