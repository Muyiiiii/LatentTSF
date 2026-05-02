#!/bin/bash
# Example: train forecaster in latent mode (with pretrained AE) or original mode.
# 1) Train an AE first (run_ae.sh) and note the checkpoint path.
# 2) Set AE_PATH to that checkpoint, then run this script.

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # or "online" / "disabled"

# Path to pretrained AE checkpoint (required for latent mode)
AE_PATH="./checkpoints/AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0/checkpoint.pth"

# Latent mode: use_latent + autoencoder_path
python -u my_train.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id Latent_DLinear \
  --model DLinear \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --step 1 \
  --use_latent \
  --encoder_type AE \
  --ae_type MLP \
  --autoencoder_path "${AE_PATH}" \
  --load_pretrained_ae 1 \
  --freeze_encoder \
  --freeze_decoder \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 64 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --patience 3 \
  --des Exp \
  --itr 1 \
  --use_lradj 1 \
  --result_csv result.csv

# Original (baseline) mode: omit --use_latent and AE-related args
# python -u my_train.py \
#   --task_name long_term_forecast --is_training 1 --model_id Ori_DLinear --model DLinear \
#   --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
#   --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
#   --d_model 512 --train_epochs 10 --batch_size 32 --learning_rate 0.001 --des Exp --itr 1
