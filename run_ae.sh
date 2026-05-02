#!/bin/bash
# Train autoencoder only (e.g. MLP or MAE). Then use the checkpoint for latent forecasting via my_train.py.

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline  # or "online" / "disabled"

# Single-dataset example: ETTh1
python -u my_AE.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_AE \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 24 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 64 \
  --train_epochs 500 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --patience 20 \
  --seed 42 \
  --use_lradj 0 \
  --ae_type MLP \
  --ae_loss MAE \
  --des Exp-sl24-lr0.0005-500-32bs \
  --itr 1

# Checkpoint will be under ./checkpoints/ (see printed setting name).
# Use that path as --autoencoder_path when running my_train.py with --use_latent.
