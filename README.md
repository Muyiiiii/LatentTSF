# LatentTSF

Latent Space Time Series Forecasting: train a forecaster in the latent space of a pretrained autoencoder (AE or MAE), with latent loss plus perceptual loss.

## Overview

- **Main loss:** Latent Loss = MSE(Z_pred, Z_y) + β × (1 - cosine_sim(Z_pred, Z_y))
- **Auxiliary loss:** Perceptual Loss = MSE(Decoder(Z_pred), Y_target)
- **Total:** Loss = Latent_Loss + α × Perceptual_Loss

**Architecture:**  
X_input → Encoder (frozen) → Z_x → TSF Model → Z_pred  
Y_target → Encoder (frozen) → Z_y  

Encoder and decoder are frozen; the TSF model is trained in latent space. Perceptual loss encourages plausible decoded waveforms.

## Main Files

| File | Description |
|------|-------------|
| `my_train.py` | Main training script (latent or original mode) |
| `my_utils.py` | Args, validation, testing, CSV logging |
| `my_AE.py` | MLP / CNN / Temporal autoencoders and training |
| `my_MAE.py` | Masked autoencoder (MAE) and training |
| `my_temporal_AE.py` | Temporal AE (seq_len dimension) and training |
| `run.py` | Original repo training entry |
| `run_train.sh` | Example: train forecaster (latent or original) |
| `run_ae.sh` | Example: train autoencoder only |

## Requirements

- Python ≥3.8
- PyTorch (install separately to match your CUDA version, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121` — see https://pytorch.org/get-started/locally/)
- All other dependencies: `pip install -r requirements.txt`

`requirements.txt` mirrors the upstream
[Time-Series-Library](https://github.com/thuml/Time-Series-Library/blob/main/requirements.txt)
list (so all baseline models in `models/` can be loaded), plus `wandb`
for experiment logging. `torch` is intentionally not pinned.

## Setup

Download the benchmark datasets to `./dataset/` (created on first run):

```bash
python download_datasets.py
```

This pulls ETTh1/2, ETTm1/2, weather, electricity, traffic, exchange_rate,
and solar_energy from HuggingFace into `./dataset/`. Users in mainland
China can speed it up by exporting a mirror endpoint **before** running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_datasets.py
```

## Quick Start

1. **Train an autoencoder** (e.g. MLP AE on ETTh1):

```bash
bash run_ae.sh
```

Or manually:

```bash
python my_AE.py --task_name long_term_forecast --is_training 1 --model_id AE_ETTh1 --model DLinear \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
  --seq_len 24 --label_len 0 --pred_len 96 --step 1 --ae_type MLP --ae_loss MAE \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 64 --train_epochs 500 --batch_size 32 \
  --learning_rate 0.0005 --patience 20 --des Exp-sl24-lr0.0005-500-32bs --itr 1 --use_lradj 0
```

> The example above reproduces the ETTh1 row in the **Pretrained AE Checkpoints** table below. For other datasets, swap in the `enc_in / d_model / d_ff` from that table and use the matching `--des` value.

2. **Train latent forecaster** (use pretrained AE):

```bash
bash run_train.sh
```

Or manually (latent mode with DLinear):

```bash
python my_train.py --task_name long_term_forecast --is_training 1 --model_id Latent_DLinear --model DLinear \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
  --seq_len 96 --label_len 0 --pred_len 96 --step 1 --use_latent \
  --encoder_type AE --ae_type MLP --autoencoder_path ./checkpoints/<AE_setting>/checkpoint.pth \
  --load_pretrained_ae 1 --freeze_encoder --freeze_decoder \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --d_ff 64 --train_epochs 10 --batch_size 32 \
  --learning_rate 0.001 --patience 3 --des Exp --itr 1 --use_lradj 1
```

3. **Original (baseline) forecaster** (no AE):

```bash
python my_train.py --task_name long_term_forecast --is_training 1 --model_id Ori_DLinear --model DLinear \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --train_epochs 10 --batch_size 32 --learning_rate 0.001 --des Exp --itr 1
```

## Pretrained AE Checkpoints

The `./checkpoints/` directory ships with the 9 pretrained MLP autoencoders used in the paper. All AEs are trained with `seq_len=24`, `ae_type=MLP`, `ae_loss=MAE`, `lradj=0`.

| Dataset | enc_in | d_model | d_ff | AE training config | Checkpoint folder |
|---|---|---|---|---|---|
| ETTh1 | 7 | 32 | 64 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| ETTh2 | 7 | 64 | 128 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_ETTh2_AE_ETTh2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| ETTm1 | 7 | 32 | 64 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_ETTm1_AE_ETTm1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| ETTm2 | 7 | 64 | 128 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_ETTm2_AE_ETTm2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| exchange_rate | 8 | 128 | 256 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_exchange_rate_AE_custom_ftM_sl24_dm128_dff256_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| weather | 21 | 64 | 128 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_weather_AE_custom_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |
| electricity | 321 | 512 | 1024 | lr=0.0001, bs=16, ep=500 | `AutoEncoder_MLP_MAE_electricity_AE_custom_ftM_sl24_dm512_dff1024_lradj0_Exp-lr0.0001-500-16bs_0` |
| traffic | 862 | 512 | 1024 | lr=0.0001, bs=16, ep=500 | `AutoEncoder_MLP_MAE_traffic_AE_custom_ftM_sl24_dm512_dff1024_lradj0_Exp-lr0.0001-500-16bs_0` |
| Solar | 137 | 256 | 512 | lr=0.0005, bs=32, ep=500 | `AutoEncoder_MLP_MAE_Solar_AE_Solar_ftM_sl24_dm256_dff512_lradj0_Exp-sl24-lr0.0005-500-32bs_0` |

Pass the corresponding folder's `checkpoint.pth` as `--autoencoder_path` when running `my_train.py` with `--use_latent`. To retrain an AE from scratch, see `run_ae.sh` (defaults match the ETTh1 config above).

## Configuration Notes

- For latent mode, **label_len** is typically 0.
- Set **--result_csv** to save metrics (e.g. `results.csv`).
- Use **--encoder_type AE** or **--encoder_type MAE**; for MAE you must train via `my_MAE.py` first and pass **--patch_len** and **--mask_ratio** consistently.
- **ae_type**: `MLP`, `MLP_REVIN`, `CNN`, `Temporal`, `TemporalCNN` (see `my_AE.py` and `my_temporal_AE.py`).

## Project Structure

```
LatentTSF/
├── my_train.py        # Main training (latent / original)
├── my_utils.py        # Args, valid, test, CSV
├── my_AE.py           # Autoencoder (MLP/CNN/Temporal) + AE training
├── my_MAE.py          # Masked autoencoder + MAE training
├── my_temporal_AE.py  # Temporal AE + training
├── run.py             # Original repo entry
├── run_train.sh       # Example train script
├── run_ae.sh          # Example AE training script
├── RevIN.py           # RevIN normalization
├── data_provider/     # Data loading
├── exp/               # Experiment management
├── layers/            # Model layers
├── models/            # TSF models (DLinear, iTransformer, etc.)
├── utils/             # Metrics, early stopping, etc.
├── checkpoints/       # Saved models
└── dataset/           # Data (create and add your datasets)
```

## License

This project is released under the [MIT License](LICENSE).

This repository also incorporates code from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
and [N-BEATS](https://github.com/ElementAI/N-BEATS). See [NOTICE.md](NOTICE.md)
for full attribution. **Note:** `utils/losses.py`, `utils/m4_summary.py`, and
`data_provider/m4.py` are licensed under **CC BY-NC 4.0 (non-commercial only)**
by Element AI Inc., and are NOT covered by the MIT license — replace or remove
them if you need to use this codebase commercially.
