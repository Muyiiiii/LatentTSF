"""
Latent Space Forecasting with Latent Loss + Perceptual Loss.

- Main loss: Latent Loss = MSE(Z_pred, Z_y) + beta * (1 - cosine_sim(Z_pred, Z_y))
- Auxiliary loss: Perceptual Loss = MSE(Decoder(Z_pred), Y_target)
- Total Loss = latent_loss + alpha * perceptual_loss

Architecture:
  X_input -> Encoder(frozen) -> Z_x -> TSF Model -> Z_pred
  Y_target -> Encoder(frozen) -> Z_y

  Latent Loss: MSE(Z_pred, Z_y) + beta * cosine_loss(Z_pred, Z_y)
  Perceptual Loss: MSE(Decoder(Z_pred), Y_target)

Features:
- Encoder and Decoder are fully frozen to preserve pretrained pairing.
- Forecasting is learned in latent space.
- Perceptual loss encourages plausible decoded waveforms.
"""

import os
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.backends
import numpy as np

import time
from torch import optim
import wandb

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from my_utils import model_dict, args_train, set_seed, acquire_device, get_data, save_result_csv
from my_AE import get_autoencoder
from my_MAE import MaskedAutoEncoder


def cosine_similarity_loss(pred_embedding, target_embedding):
    """
    计算预测embedding和目标embedding之间的cosine similarity loss
    返回 1 - mean(cosine_similarity)，范围 [0, 2]
    """
    pred_flat = pred_embedding.reshape(-1, pred_embedding.shape[-1])
    target_flat = target_embedding.reshape(-1, target_embedding.shape[-1])
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    loss = 1 - cos_sim.mean()

    # loss_calc_func = nn.CosineSimilarity(eps=1e-8, dim=1)
    # loss = (1 - (loss_calc_func(pred_embedding, target_embedding.detach()).mean()))
    
    return loss


def build_dec_inp_latent(z_x, label_len, pred_len, d_model, device):
    """
    Build decoder input for latent mode.
    dec_inp: [label_len + pred_len]; first label_len from end of z_x, rest zeros.
    """
    batch_size = z_x.size(0)
    dec_inp_pred = torch.zeros(batch_size, pred_len, d_model).float().to(device)
    if label_len > 0:
        dec_inp_label = z_x[:, -label_len:, :]
        return torch.cat([dec_inp_label, dec_inp_pred], dim=1)
    return dec_inp_pred


def build_dec_inp_original(batch_y, label_len, pred_len, device):
    """
    构建 original mode 的 decoder input
    dec_inp: [label_len + pred_len], 前label_len来自batch_y，后pred_len为零
    """
    batch_size = batch_y.size(0)
    c_dim = batch_y.size(-1)
    dec_inp_pred = torch.zeros(batch_size, pred_len, c_dim).float().to(device)
    if label_len > 0:
        dec_inp_label = batch_y[:, :label_len, :]
        return torch.cat([dec_inp_label, dec_inp_pred], dim=1)
    return dec_inp_pred


def build_marks(batch_size, seq_len, mark_dim, device):
    """Build zero-filled marks with explicit length."""
    return torch.zeros(batch_size, seq_len, mark_dim).float().to(device)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def encoder_encode(encoder, x, encoder_type, latent_norm=None):
    enc = unwrap(encoder)
    latent = enc.encode_seq(x) if encoder_type == 'MAE' else enc.encode(x)
    if latent_norm is not None:
        latent = latent_norm(latent)
    return latent

def encoder_decode(encoder, latent, encoder_type):
    enc = unwrap(encoder)
    return enc.decode_seq(latent) if encoder_type == 'MAE' else enc.decode(latent)



def compute_latent_loss(z_pred, z_target, mse_weight=1.0, cosine_weight=1.0):
    """
    计算 Latent Loss = MSE + cosine_loss
    z_pred: 预测的 latent, shape (batch, seq_len, d_model)
    z_target: 目标的 latent, shape (batch, seq_len, d_model)
    """
    mse_loss = F.mse_loss(z_pred, z_target)
    cos_loss = cosine_similarity_loss(z_pred, z_target)
    
    latent_loss = mse_weight * mse_loss + cosine_weight * cos_loss
    return latent_loss, mse_loss, cos_loss


def valid_v20(args, model, valid_loader, criterion, device, encoder=None, encoder_type='AE',
              perceptual_weight=0.1, mse_weight=1.0, cosine_weight=1.0, latent_norm=None):
    """Validation function for latent forecasting."""
    model.eval()
    if encoder is not None:
        encoder.eval()

    total_loss = []
    total_latent_loss = []
    total_perceptual_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if encoder is not None:
                # Encode: 只对 z_x 做 LN (TSF 输入侧)，z_y 保持原始 latent 空间
                z_x = encoder_encode(encoder, batch_x, encoder_type, latent_norm)
                batch_y_target = batch_y[:, -args.pred_len:, :]
                z_y = encoder_encode(encoder, batch_y_target, encoder_type, None)

                # Build decoder input
                dec_inp = build_dec_inp_latent(z_x, args.label_len, args.pred_len, args.d_model, device)

                # Predict latent (z_pred not normalized to match z_y)
                z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)
                z_pred = z_pred[:, -args.pred_len:, :]

                # Latent Loss
                latent_loss, _, _ = compute_latent_loss(z_pred, z_y, mse_weight=mse_weight, cosine_weight=cosine_weight)

                # Perceptual Loss (Decoder output vs target)
                y_pred = encoder_decode(encoder, z_pred, encoder_type)
                f_dim = -1 if args.features == 'MS' else 0
                y_pred = y_pred[:, :, f_dim:]
                batch_y_target = batch_y_target[:, :, f_dim:]
                perceptual_loss = criterion(y_pred, batch_y_target)

                # Total Loss
                # loss = latent_loss + perceptual_weight * perceptual_loss
                
                # Validation metric: only use decoded output criterion (same as valid_original)
                loss = perceptual_loss

                total_loss.append(loss.item())
                total_latent_loss.append(latent_loss.item())
                total_perceptual_loss.append(perceptual_loss.item())

    avg_loss = np.average(total_loss)
    model.train()
    return avg_loss


def test_v20(args, model, test_data, test_loader, setting, device, encoder=None, encoder_type='AE', test=0, latent_norm=None):
    """
    V20 测试函数
    """
    if test:
        print(f"loading model from {args.checkpoints + setting + '/' + 'checkpoint.pth'}")
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    if encoder is not None:
        encoder.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            if encoder is not None:
                # Encode: apply LN only to z_x (TSF input)
                z_x = encoder_encode(encoder, batch_x, encoder_type, latent_norm)
                dec_inp = build_dec_inp_latent(z_x, args.label_len, args.pred_len, args.d_model, device)

                # z_pred not normalized so decode stays in raw latent space
                z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)
                z_pred = z_pred[:, -args.pred_len:, :]
                outputs = encoder_decode(encoder, z_pred, encoder_type)
            else:
                # Original mode
                dec_inp = build_dec_inp_original(batch_y, args.label_len, args.pred_len, device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if test_data.scale and args.inverse:
                shape = batch_y.shape
                if outputs.shape[-1] != batch_y.shape[-1]:
                    outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            pred = outputs
            true = batch_y
            preds.append(pred)
            trues.append(true)

            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     if test_data.scale and args.inverse:
            #         shape = input.shape
            #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)

    folder_path = './result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}")

    result_file = f"result/result_latent_forecast_v20_{encoder_type}.txt"

    with open(result_file, 'a') as f:
        f.write(setting + "  \n")
        f.write(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}")
        f.write('\n\n')

    return mae, mse, rmse, mape, mspe


def valid_original(args, model, valid_loader, criterion, device):
    """Original mode 验证函数"""
    model.eval()
    total_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = build_dec_inp_original(batch_y, args.label_len, args.pred_len, device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return np.average(total_loss)


def test_original(args, model, test_data, test_loader, setting, device, test=0):
    """Test function for original (non-latent) mode."""
    if test:
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = build_dec_inp_original(batch_y, args.label_len, args.pred_len, device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if test_data.scale and args.inverse:
                shape = batch_y.shape
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            preds.append(outputs)
            trues.append(batch_y)

            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     if test_data.scale and args.inverse:
            #         shape = input.shape
            #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
            #     gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}")

    with open("result/result_original_forecast_v20.txt", 'a') as f:
        f.write(setting + "  \n")
        f.write(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}\n\n")

    return mae, mse, rmse, mape, mspe


if __name__ == "__main__":
    args = args_train()

    # 获取参数
    encoder_type = getattr(args, 'encoder_type', 'AE')
    perceptual_weight = getattr(args, 'perceptual_weight', 0.1)
    mse_weight = getattr(args, 'mse_weight', 1.0)
    cosine_weight = getattr(args, 'cosine_weight', 1.0)
    reconstruction_weight = getattr(args, 'reconstruction_weight', 1.0)
    use_latent_norm = getattr(args, 'use_latent_norm', True)
    ae_loss_type = getattr(args, 'ae_loss', 'MSE').upper()
    use_lradj_value = getattr(args, 'use_lradj', 1)

    for iter in range(args.itr):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<")

        set_seed(args.seed)
        device = acquire_device(args)

        # Load data
        train_data, train_loader = get_data(args, flag='train')
        valid_data, valid_loader = get_data(args, flag='val')
        test_data, test_loader = get_data(args, flag='test')

        encoder = None

        if args.use_latent:
            # ============ Latent Mode ============
            print(f"V20: Latent Space Forecasting")
            print(f"Encoder type: {encoder_type}")
            print(f"Perceptual weight (α): {perceptual_weight}")
            print(f"MSE weight: {mse_weight}")
            print(f"Cosine weight (β): {cosine_weight}")
            print(f"Reconstruction weight (γ): {reconstruction_weight}")
            print(f"Loss = Latent_Loss + {perceptual_weight} * Perceptual_Loss + {reconstruction_weight} * Reconstruction_Loss")
            print(f"Latent_Loss = {mse_weight} * MSE + {cosine_weight} * Cosine")

            setting = f"LatentV20_{encoder_type}_{ae_loss_type}_{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_st{args.step}_batch{args.batch_size}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_lr{args.learning_rate}_pw{perceptual_weight}_mw{mse_weight}_cw{cosine_weight}_rw{reconstruction_weight}_lradj{use_lradj_value}_{args.des}_{iter}"

            # 创建 Encoder
            ae_type = getattr(args, 'ae_type', 'MLP').upper()
            load_pretrained_ae = getattr(args, 'load_pretrained_ae', True)

            if encoder_type == 'MAE':
                encoder = MaskedAutoEncoder(args).float()
            else:
                # Use get_autoencoder to select MLP or CNN by ae_type
                encoder = get_autoencoder(args).float()

            if args.use_multi_gpu and args.use_gpu:
                encoder = nn.DataParallel(encoder, device_ids=args.device_ids)
            encoder.to(device)

            # 根据 load_pretrained_ae 决定是否加载预训练权重
            if load_pretrained_ae:
                if not args.autoencoder_path or not os.path.exists(args.autoencoder_path):
                    raise ValueError(f"V20 requires pretrained {encoder_type}. Please provide --autoencoder_path")
                print(f"Loading pretrained {encoder_type} (ae_type={ae_type}) from {args.autoencoder_path}")
                encoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
            else:
                print(f"Using randomly initialized {encoder_type} (ae_type={ae_type}) - no pretrained weights loaded")

            # Freeze encoder and/or decoder
            freeze_encoder = getattr(args, 'freeze_encoder', True)
            freeze_decoder = getattr(args, 'freeze_decoder', True)

            if freeze_encoder and freeze_decoder:
                print(f"Freezing {encoder_type} (encoder + decoder)")
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()
            else:
                # 分别控制 encoder 和 decoder
                enc_unwrapped = unwrap(encoder)

                if freeze_encoder:
                    print(f"Freezing {encoder_type} encoder only")
                    if encoder_type == 'MAE':
                        for param in enc_unwrapped.encoder.parameters():
                            param.requires_grad = False
                    else:  # AE
                        for param in enc_unwrapped.encoder.parameters():
                            param.requires_grad = False
                else:
                    print(f"{encoder_type} encoder is trainable")

                if freeze_decoder:
                    print(f"Freezing {encoder_type} decoder only")
                    if encoder_type == 'MAE':
                        for param in enc_unwrapped.decoder.parameters():
                            param.requires_grad = False
                    else:  # AE
                        for param in enc_unwrapped.decoder.parameters():
                            param.requires_grad = False
                else:
                    print(f"{encoder_type} decoder is trainable")

                # Train mode if both trainable; else keep eval for frozen parts
                if not freeze_encoder and not freeze_decoder:
                    encoder.train()
                else:
                    encoder.eval()

            # 创建预测模型 - latent 维度
            original_enc_in = args.enc_in
            original_dec_in = args.dec_in
            original_c_out = args.c_out

            args.enc_in = args.d_model
            args.dec_in = args.d_model
            args.c_out = args.d_model

            model = model_dict[args.model].Model(args).float()

            args.enc_in = original_enc_in
            args.dec_in = original_dec_in
            args.c_out = original_c_out

            # Latent LayerNorm (fixed, no learnable params)
            if use_latent_norm:
                print(f"Using LayerNorm on latent space (z_x, z_pred) - fixed, no learnable params")
                latent_norm = nn.LayerNorm(args.d_model, elementwise_affine=False).to(device)
            else:
                latent_norm = None
        else:
            # ============ Original Mode ============
            print(f"V20: Original Mode (Baseline)")
            setting = f"OriginalV20_{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_st{args.step}_batch{args.batch_size}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_lr{args.learning_rate}_lradj{use_lradj_value}_{args.des}_{iter}"

            model = model_dict[args.model].Model(args).float()
            latent_norm = None

        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # Initialize wandb
        wandb_config = {
            "task": args.task_name,
            "model": args.model,
            "dataset": args.data_path.split('.')[0],
            "features": args.features,
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "pred_len": args.pred_len,
            "dataset_step": args.step,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "n_heads": args.n_heads,
            "e_layers": args.e_layers,
            "d_layers": args.d_layers,
            "enc_in": args.enc_in,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_epochs": args.train_epochs,
            "patience": args.patience,
            "seed": args.seed,
            "iteration": iter,
            "use_lradj": use_lradj_value,
        }

        # Add experiment label if provided
        if args.exp_label:
            wandb_config["exp_label"] = args.exp_label

        if args.use_latent:
            # Get encoder/decoder freeze status and learning rates for wandb logging
            _freeze_encoder = getattr(args, 'freeze_encoder', True)
            _freeze_decoder = getattr(args, 'freeze_decoder', True)
            _encoder_lr = getattr(args, 'encoder_lr', args.learning_rate) if not _freeze_encoder else None
            _decoder_lr = getattr(args, 'decoder_lr', args.learning_rate) if not _freeze_decoder else None

            wandb_config.update({
                "mode": "Latent",
                "encoder_type": encoder_type,
                "ae_type": ae_type,
                "ae_loss": ae_loss_type,
                "perceptual_weight": perceptual_weight,
                "mse_weight": mse_weight,
                "cosine_weight": cosine_weight,
                "reconstruction_weight": reconstruction_weight,
                "use_latent_norm": use_latent_norm,
                "freeze_encoder": _freeze_encoder,
                "freeze_decoder": _freeze_decoder,
                "encoder_lr": _encoder_lr,
                "decoder_lr": _decoder_lr,
            })
            default_project_name = "LatentTSF-V20-Latent"
        else:
            wandb_config.update({
                "mode": "Original",
            })
            default_project_name = "LatentTSF-V20-Original"

        # Use environment variable WANDB_PROJECT if set, otherwise use default
        wandb_project = os.environ.get('WANDB_PROJECT', default_project_name)
        print(f"WandB Project: {wandb_project}")
        print(f"  - Environment WANDB_PROJECT: {os.environ.get('WANDB_PROJECT', 'NOT SET')}")
        print(f"  - Default project name: {default_project_name}")

        wandb.init(
            project=wandb_project,
            name=setting,
            config=wandb_config,
            settings=wandb.Settings(start_method="thread")
        )

        if args.use_multi_gpu and args.use_gpu:
            print('Using Multi-GPU')
            model = nn.DataParallel(model, device_ids=args.device_ids)
        model.to(device)

        # 重置显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        training_start_time = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        # Optimizer with optional separate encoder/decoder learning rates
        if args.use_latent and not (freeze_encoder and freeze_decoder):
            param_groups = []

            # TSF model (main)
            model_params = list(model.parameters())
            tsf_lr = args.learning_rate
            param_groups.append({'params': model_params, 'lr': tsf_lr})
            print(f"TSF Model learning rate: {tsf_lr}")

            # Encoder 参数
            if not freeze_encoder:
                enc_unwrapped = unwrap(encoder)
                if encoder_type == 'MAE':
                    encoder_params = list(enc_unwrapped.encoder.parameters())
                else:  # AE
                    encoder_params = list(enc_unwrapped.encoder.parameters())

                encoder_lr = getattr(args, 'encoder_lr', args.learning_rate)
                param_groups.append({'params': encoder_params, 'lr': encoder_lr})
                print(f"Encoder learning rate: {encoder_lr}")

            # Decoder params
            if not freeze_decoder:
                enc_unwrapped = unwrap(encoder)
                if encoder_type == 'MAE':
                    decoder_params = list(enc_unwrapped.decoder.parameters())
                else:  # AE
                    decoder_params = list(enc_unwrapped.decoder.parameters())

                decoder_lr = getattr(args, 'decoder_lr', args.learning_rate)
                param_groups.append({'params': decoder_params, 'lr': decoder_lr})
                print(f"Decoder learning rate: {decoder_lr}")

            model_optim = optim.Adam(param_groups)
        else:
            # 标准模式或完全冻结 encoder/decoder
            model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
            print(f"Model learning rate: {args.learning_rate}")

        criterion = nn.MSELoss()

        # Gradient accumulation
        accum_steps = getattr(args, 'accum_steps', 1)
        if accum_steps > 1:
            print(f"Using gradient accumulation: {accum_steps} steps (effective batch = {args.batch_size * accum_steps})")

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        actual_epochs = 0

        for epoch in range(args.train_epochs):
            actual_epochs = epoch + 1
            iter_count = 0
            train_loss = []
            train_latent_loss = []
            train_mse_loss = []
            train_cos_loss = []
            train_perceptual_loss = []
            train_reconstruction_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # Zero grad at the start of accumulation window
                if i % accum_steps == 0:
                    model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                if args.use_latent and encoder is not None:
                    # ============ Latent Mode ============
                    # Encode input
                    # Apply LN only to z_x (TSF input); z_y in raw latent space
                    z_x = encoder_encode(encoder, batch_x, encoder_type, latent_norm)
                    batch_y_target = batch_y[:, -args.pred_len:, :]
                    z_y = encoder_encode(encoder, batch_y_target, encoder_type, None)

                    # Detach embeddings to prevent gradients flowing back to encoder
                    z_x = z_x
                    z_y = z_y

                    # 构建 decoder input
                    dec_inp = build_dec_inp_latent(z_x, args.label_len, args.pred_len, args.d_model, device)

                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)
                            z_pred = z_pred[:, -args.pred_len:, :]
                            # z_pred not normalized to match z_y

                            latent_loss, mse_loss, cos_loss = compute_latent_loss(
                                z_pred, z_y, mse_weight=mse_weight, cosine_weight=cosine_weight
                            )

                            # Detach z_pred before decoder to prevent gradients flowing back to TSF model
                            y_pred = encoder_decode(encoder, z_pred, encoder_type)
                            f_dim = -1 if args.features == 'MS' else 0
                            perceptual_loss = criterion(y_pred[:, :, f_dim:], batch_y_target[:, :, f_dim:])

                            # Reconstruction loss: ensure z_y can be decoded back to y_target
                            y_recon = encoder_decode(encoder, z_y, encoder_type)
                            reconstruction_loss = F.l1_loss(y_recon[:, :, f_dim:], batch_y_target[:, :, f_dim:])

                            loss = latent_loss + perceptual_weight * perceptual_loss + reconstruction_weight * reconstruction_loss
                    else:
                        z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)
                        z_pred = z_pred[:, -args.pred_len:, :]
                        # z_pred 不做 LN，与 z_y 保持一致

                        latent_loss, mse_loss, cos_loss = compute_latent_loss(
                            z_pred, z_y, mse_weight=mse_weight, cosine_weight=cosine_weight
                        )

                        # Detach z_pred before decoder to prevent gradients flowing back to TSF model
                        y_pred = encoder_decode(encoder, z_pred, encoder_type)
                        f_dim = -1 if args.features == 'MS' else 0
                        perceptual_loss = criterion(y_pred[:, :, f_dim:], batch_y_target[:, :, f_dim:])

                        # Reconstruction loss: ensure z_y can be decoded back to y_target
                        y_recon = encoder_decode(encoder, z_y, encoder_type)
                        reconstruction_loss = F.l1_loss(y_recon[:, :, f_dim:], batch_y_target[:, :, f_dim:])

                        loss = latent_loss + perceptual_weight * perceptual_loss + reconstruction_weight * reconstruction_loss

                    train_loss.append(loss.item())
                    train_latent_loss.append(latent_loss.item())
                    train_mse_loss.append(mse_loss.item())
                    train_cos_loss.append(cos_loss.item())
                    train_perceptual_loss.append(perceptual_loss.item())
                    train_reconstruction_loss.append(reconstruction_loss.item())
                else:
                    # ============ Original Mode ============
                    dec_inp = build_dec_inp_original(batch_y, args.label_len, args.pred_len, device)

                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if args.features == 'MS' else 0
                            outputs = outputs[:, -args.pred_len:, f_dim:]
                            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y_target)
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y_target = batch_y[:, -args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y_target)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if args.use_latent:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                        print(f"\t  latent: {latent_loss.item():.7f} (mse: {mse_loss.item():.7f}, cos: {cos_loss.item():.7f})")
                        print(f"\t  perceptual: {perceptual_loss.item():.7f}, recon: {reconstruction_loss.item():.7f}")
                    else:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                # Scale loss for gradient accumulation
                loss_scaled = loss / accum_steps

                if args.use_amp:
                    scaler.scale(loss_scaled).backward()
                    # Only step when accumulation is complete
                    if (i + 1) % accum_steps == 0:
                        scaler.step(model_optim)
                        scaler.update()
                else:
                    loss_scaled.backward()
                    # Only step when accumulation is complete
                    if (i + 1) % accum_steps == 0:
                        model_optim.step()

            # Handle remaining gradients at epoch end (incomplete accumulation window)
            if train_steps % accum_steps != 0:
                if args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    model_optim.step()

            print(f"Epoch: {epoch+1}, cost time: {time.time()-epoch_time:.3f}s")

            train_loss_avg = np.average(train_loss)
            if args.use_latent:
                train_latent_avg = np.average(train_latent_loss)
                train_mse_avg = np.average(train_mse_loss)
                train_cos_avg = np.average(train_cos_loss)
                train_perceptual_avg = np.average(train_perceptual_loss)
                train_reconstruction_avg = np.average(train_reconstruction_loss)
                print(f"Train Loss: {train_loss_avg:.7f}")
                print(f"  Latent: {train_latent_avg:.7f} (MSE: {train_mse_avg:.7f}, Cos: {train_cos_avg:.7f})")
                print(f"  Perceptual: {train_perceptual_avg:.7f}, Reconstruction: {train_reconstruction_avg:.7f}")
            else:
                print(f"Train Loss: {train_loss_avg:.7f}")

            if args.use_latent:
                valid_loss = valid_v20(args, model, valid_loader, criterion, device, encoder, encoder_type,
                                       perceptual_weight, mse_weight, cosine_weight, latent_norm)
            else:
                valid_loss = valid_original(args, model, valid_loader, criterion, device)

            print(f"Epoch: {epoch+1}, Valid Loss: {valid_loss:.7f}")

            # Log to wandb
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss_avg,
                "valid_loss": valid_loss,
                "learning_rate": model_optim.param_groups[0]['lr']
            }

            if args.use_latent:
                wandb_log.update({
                    "train_latent_loss": train_latent_avg,
                    "train_mse_loss": train_mse_avg,
                    "train_cos_loss": train_cos_avg,
                    "train_perceptual_loss": train_perceptual_avg,
                    "train_reconstruction_loss": train_reconstruction_avg,
                })

            wandb.log(wandb_log)

            early_stopping(valid_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rate if enabled
            use_lradj = getattr(args, 'use_lradj', 1)
            if use_lradj:
                adjust_learning_rate(model_optim, epoch+1, args)

        # Load best model
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        # Total training time
        training_time = time.time() - training_start_time
        print(f"Total training time: {training_time:.2f}s ({training_time/60:.2f}min)")

        print(f">>>>>>>start testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if args.use_latent:
            mae, mse, rmse, mape, mspe = test_v20(args, model, test_data, test_loader, setting, device,
                                                   encoder, encoder_type, test=0, latent_norm=latent_norm)
        else:
            mae, mse, rmse, mape, mspe = test_original(args, model, test_data, test_loader, setting, device, test=0)

        # Log final test metrics to wandb
        wandb.log({
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mape": mape,
            "test_mspe": mspe,
        })

        # 获取最大显存使用量
        if torch.cuda.is_available():
            max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            max_memory_mb = None

        # Log training stats to wandb
        wandb.log({
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "actual_epochs": actual_epochs,
        })

        if max_memory_mb is not None:
            wandb.log({"max_memory_mb": max_memory_mb})

        # Save results to CSV (latent: pass weights; original: pass None)
        if args.use_latent:
            save_result_csv(args, setting, is_latent=True, mae=mae, mse=mse, rmse=rmse, mape=mape, mspe=mspe,
                           actual_epochs=actual_epochs,
                           perceptual_weight=perceptual_weight, mse_weight=mse_weight, cosine_weight=cosine_weight,
                           reconstruction_weight=reconstruction_weight,
                           record_seq_pred_len=True, max_memory_mb=max_memory_mb, training_time=training_time//60,
                           use_lradj=use_lradj_value)
        else:
            save_result_csv(args, setting, is_latent=False, mae=mae, mse=mse, rmse=rmse, mape=mape, mspe=mspe,
                           actual_epochs=actual_epochs,
                           perceptual_weight=None, mse_weight=None, cosine_weight=None,
                           reconstruction_weight=None,
                           record_seq_pred_len=True, max_memory_mb=max_memory_mb, training_time=training_time//60,
                           use_lradj=use_lradj_value)

        # 删除 checkpoint（如果设置了 delete_checkpoint）
        if args.delete_checkpoint:
            checkpoint_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Deleted checkpoint: {checkpoint_path}")
            # Remove empty checkpoint dir if possible
            try:
                if os.path.exists(path) and not os.listdir(path):
                    os.rmdir(path)
                    print(f"Deleted empty checkpoint directory: {path}")
            except OSError:
                pass  # Dir not empty, keep it

        # Finish wandb run
        wandb.finish()

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
