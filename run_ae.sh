#!/bin/bash
# Train one or more autoencoders from scratch.
# Set `datasets` at the top to either a single name (quick test) or all 9
# (reproduce every shipped checkpoint).
#
# All AEs in ./checkpoints/ were trained with this script and:
#   - seq_len = 24, ae_type = MLP, ae_loss = MAE, lradj disabled
#   - 500 epochs (early-stopped via --patience 20), seed 42
# Per-dataset (d_model, d_ff, learning_rate, batch_size, des) values below
# match the README "Pretrained AE Checkpoints" table; retraining overwrites
# the shipped checkpoints in place.

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # or "online" / "disabled"

# ============ User-tunable ============
# Space-separated list. One quick example or all 9.
datasets="ETTh1"
# datasets="ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity traffic Solar"
# ======================================

# Per-dataset config (matches README checkpoint table).
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

declare -A d_model          ; d_model["ETTh1"]=32   ; d_model["ETTh2"]=64
                              d_model["ETTm1"]=32   ; d_model["ETTm2"]=64
                              d_model["weather"]=64 ; d_model["exchange_rate"]=128
                              d_model["electricity"]=512 ; d_model["traffic"]=512
                              d_model["Solar"]=256

declare -A d_ff             ; d_ff["ETTh1"]=64   ; d_ff["ETTh2"]=128
                              d_ff["ETTm1"]=64   ; d_ff["ETTm2"]=128
                              d_ff["weather"]=128 ; d_ff["exchange_rate"]=256
                              d_ff["electricity"]=1024 ; d_ff["traffic"]=1024
                              d_ff["Solar"]=512

declare -A learning_rate    ; learning_rate["ETTh1"]=0.0005 ; learning_rate["ETTh2"]=0.0005
                              learning_rate["ETTm1"]=0.0005 ; learning_rate["ETTm2"]=0.0005
                              learning_rate["weather"]=0.0005 ; learning_rate["exchange_rate"]=0.0005
                              learning_rate["electricity"]=0.0001 ; learning_rate["traffic"]=0.0001
                              learning_rate["Solar"]=0.0005

declare -A batch_size       ; batch_size["ETTh1"]=32 ; batch_size["ETTh2"]=32
                              batch_size["ETTm1"]=32 ; batch_size["ETTm2"]=32
                              batch_size["weather"]=32 ; batch_size["exchange_rate"]=32
                              batch_size["electricity"]=16 ; batch_size["traffic"]=16
                              batch_size["Solar"]=32

# des suffix tags the checkpoint folder; matches the shipped checkpoints so
# retraining overwrites them in place. Two recipes are used (small datasets
# use bs=32/lr=5e-4, large ones use bs=16/lr=1e-4).
declare -A des              ; des["ETTh1"]="Exp-sl24-lr0.0005-500-32bs"
                              des["ETTh2"]="Exp-sl24-lr0.0005-500-32bs"
                              des["ETTm1"]="Exp-sl24-lr0.0005-500-32bs"
                              des["ETTm2"]="Exp-sl24-lr0.0005-500-32bs"
                              des["weather"]="Exp-sl24-lr0.0005-500-32bs"
                              des["exchange_rate"]="Exp-sl24-lr0.0005-500-32bs"
                              des["electricity"]="Exp-lr0.0001-500-16bs"
                              des["traffic"]="Exp-lr0.0001-500-16bs"
                              des["Solar"]="Exp-sl24-lr0.0005-500-32bs"

for dataset in $datasets; do
  if [ -z "${data_path[$dataset]}" ]; then
    echo "[skip] unknown dataset '$dataset' (valid: ${!data_path[*]})"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "Training AutoEncoder (MLP) for: $dataset"
  echo "  enc_in=${enc_in[$dataset]}  d_model/d_ff=${d_model[$dataset]}/${d_ff[$dataset]}"
  echo "  lr=${learning_rate[$dataset]}  batch=${batch_size[$dataset]}"
  echo "================================================================"

  python -u my_AE.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${root_path[$dataset]}" \
    --data_path "${data_path[$dataset]}" \
    --model_id "${dataset}_AE" \
    --model DLinear \
    --data "${data_name[$dataset]}" \
    --features M \
    --seq_len 24 \
    --label_len 0 \
    --pred_len 24 \
    --enc_in "${enc_in[$dataset]}" \
    --dec_in "${enc_in[$dataset]}" \
    --c_out "${enc_in[$dataset]}" \
    --d_model "${d_model[$dataset]}" \
    --d_ff "${d_ff[$dataset]}" \
    --train_epochs 500 \
    --batch_size "${batch_size[$dataset]}" \
    --learning_rate "${learning_rate[$dataset]}" \
    --patience 20 \
    --seed 42 \
    --use_lradj 0 \
    --ae_type MLP \
    --ae_loss MAE \
    --des "${des[$dataset]}" \
    --itr 1
done

# Checkpoints land under
#   ./checkpoints/AutoEncoder_MLP_MAE_${dataset}_${data_name}_AE_${data_name}_ftM_sl24_dm${d_model}_dff${d_ff}_lradj0_${des}_0/checkpoint.pth
# which is the path that run_train.sh / run_train_all.sh expect.
