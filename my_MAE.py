"""
Masked Autoencoder (MAE) for Time Series Forecasting.
Inspired by Kaiming He's MAE (https://arxiv.org/abs/2111.06377).

Design:
1. Patch embedding: split time series into patches.
2. Random masking: mask a fraction of patches (e.g. 75%).
3. Encoder: 2-layer MLP.
4. Decoder: 2-layer MLP to reconstruct masked patches.
5. Loss: reconstruction loss on masked patches only.
6. No CLS token: encode() returns all patch embeddings for downstream forecasting.
"""

import os
import math
import torch
import torch.nn as nn
import torch.backends
import numpy as np
import time
from torch import optim

from utils.tools import EarlyStopping, adjust_learning_rate
from my_utils import args_train, set_seed, acquire_device, get_data


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    生成1D正弦余弦位置编码
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = np.arange(length, dtype=np.float64)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class MaskedAutoEncoder(nn.Module):
    """
    Masked Autoencoder for Time Series (2-layer MLP, no CLS token).

    - Patch-based; encoder/decoder are 2-layer MLPs.
    - MAE masking: loss only on masked patches.
    - encode() returns all patch embeddings for downstream forecasting.
    """
    def __init__(self, args):
        super().__init__()

        # 基础配置
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.patch_len = args.patch_len
        self.mask_ratio = args.mask_ratio
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_patches = self.seq_len // self.patch_len
        self.patch_dim = self.patch_len * self.enc_in

        # Position embedding (fixed sine-cosine)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.d_model),
            requires_grad=False
        )

        # Encoder: 2层MLP (patch_dim -> d_ff -> d_model)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_dim, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.GELU(),
        )

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Decoder: 2-layer MLP (d_model -> d_ff -> patch_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.patch_dim),
        )

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        # 位置编码初始化
        pos_embed = get_1d_sincos_pos_embed(self.d_model, self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, x):
        """
        将输入序列分成patches
        x: (batch, seq_len, enc_in)
        return: (batch, num_patches, patch_dim)
        """
        B, L, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_len, C)
        x = x.reshape(B, self.num_patches, -1)
        return x

    def unpatchify(self, x):
        """
        Restore patches to sequence.
        x: (batch, num_patches, patch_dim) -> (batch, seq_len, enc_in)
        """
        B = x.shape[0]
        x = x.reshape(B, self.num_patches, self.patch_len, self.enc_in)
        x = x.reshape(B, self.seq_len, self.enc_in)
        return x

    def random_masking(self, x):
        """
        随机masking
        x: (batch, num_patches, d_model)

        Returns:
        - x_masked: visible patches only
        - mask: 二值mask, 0保留, 1mask
        - ids_restore: 恢复顺序的索引
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # 随机噪声
        noise = torch.rand(B, N, device=x.device)

        # 排序
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留前len_keep个
        ids_keep = ids_shuffle[:, :len_keep]

        # 收集visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio=None):
        """Encoder forward with masking. x: (batch, seq_len, enc_in)."""
        # Patchify
        x = self.patchify(x)  # (B, num_patches, patch_dim)

        # Encode each patch
        x = self.encoder(x)  # (B, num_patches, d_model)

        # Add position embedding
        x = x + self.pos_embed

        # Random masking
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decoder前向传播
        x: visible patches (batch, num_visible, d_model)
        ids_restore: 恢复顺序的索引
        """
        B = x.shape[0]
        num_visible = x.shape[1]
        num_masked = self.num_patches - num_visible

        # 添加mask tokens
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)

        # 恢复原始顺序
        x_full = torch.gather(x_full, dim=1,
                              index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))

        # Decode
        pred = self.decoder(x_full)  # (B, num_patches, patch_dim)

        return pred

    def forward_loss(self, x, pred, mask):
        """
        Loss on masked patches only.
        x: input (batch, seq_len, enc_in); pred: reconstructed patches; mask: 1 = masked.
        """
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, x, mask_ratio=None):
        """
        完整前向传播 (training with masking)
        x: (batch, seq_len, enc_in)

        Returns:
        - loss: masked patches的重建损失
        - pred: 重建的patches
        - mask: 二值mask
        """
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)

        return loss, pred, mask

    def encode(self, x):
        """
        Latent representation without masking (for downstream forecasting).
        x: (batch, seq_len, enc_in) -> (batch, num_patches, d_model).
        """
        # Patchify
        x = self.patchify(x)  # (B, num_patches, patch_dim)

        # Encode
        x = self.encoder(x)  # (B, num_patches, d_model)

        # Add position embedding
        x = x + self.pos_embed

        return x

    def decode(self, latent):
        """
        从latent解码回原始空间
        latent: (batch, num_patches, d_model)
        Returns: (batch, seq_len, enc_in)
        """
        # Decode
        pred = self.decoder(latent)  # (B, num_patches, patch_dim)

        # Unpatchify
        pred = self.unpatchify(pred)  # (B, seq_len, enc_in)

        return pred

    def reconstruct(self, x):
        """Full reconstruction (no masking)."""
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon

    def encode_seq(self, x):
        """
        获取与AE统一接口的latent representation
        x: (batch, seq_len, enc_in)
        Returns: (batch, seq_len, d_model) - 与AE的encode接口一致

        实现：将每个patch的embedding重复patch_len次
        """
        # 先获取patch-level embedding: (B, num_patches, d_model)
        patch_embed = self.encode(x)

        # 将每个patch的embedding重复patch_len次: (B, seq_len, d_model)
        # (B, num_patches, d_model) -> (B, num_patches, patch_len, d_model) -> (B, seq_len, d_model)
        B = patch_embed.shape[0]
        seq_embed = patch_embed.unsqueeze(2).repeat(1, 1, self.patch_len, 1)
        seq_embed = seq_embed.reshape(B, self.seq_len, self.d_model)

        return seq_embed

    def decode_seq(self, latent):
        """
        Decode seq-level latent to original space (same interface as AE).
        latent: (batch, seq_len, d_model) -> (batch, seq_len, enc_in).
        Reshape to patches, average per patch, then decode.
        """
        B = latent.shape[0]
        latent = latent.reshape(B, self.num_patches, self.patch_len, self.d_model)
        patch_latent = latent.mean(dim=2)
        output = self.decode(patch_latent)

        return output


def valid_mae(args, model, valid_loader, device):
    """验证MAE (masked loss)"""
    model.eval()
    total_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss, pred, mask = model(batch_x)
            else:
                loss, pred, mask = model(batch_x)

            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def valid_mae_reconstruction(args, model, valid_loader, device):
    """Evaluate full reconstruction."""
    model.eval()
    total_loss = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)

            pred = model.reconstruct(batch_x)
            loss = criterion(pred, batch_x)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


if __name__ == "__main__":
    args = args_train()

    # Validate patch config
    assert args.seq_len % args.patch_len == 0, \
        f"seq_len ({args.seq_len}) must be divisible by patch_len ({args.patch_len})"

    for iter in range(args.itr):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<")

        set_seed(args.seed)

        setting = f"MAE_{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_pl{args.patch_len}_mr{args.mask_ratio}_dm{args.d_model}_dff{args.d_ff}_{args.des}_{iter}"

        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")

        device = acquire_device(args)

        model = MaskedAutoEncoder(args).float()
        if args.use_multi_gpu and args.use_gpu:
            print('Using Multi-GPU')
            model = nn.DataParallel(model, device_ids=args.device_ids)
        model.to(device)

        train_data, train_loader = get_data(args, flag='train')
        valid_data, valid_loader = get_data(args, flag='val')
        test_data, test_loader = get_data(args, flag='test')

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)

        # Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=args.train_epochs * train_steps,
            eta_min=1e-6
        )

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)

                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, pred, mask = model(batch_x)
                        train_loss.append(loss.item())
                else:
                    loss, pred, mask = model(batch_x)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                scheduler.step()

            print(f"Epoch: {epoch+1}, cost time: {time.time()-epoch_time:.3f}s")
            train_loss = np.average(train_loss)
            valid_loss = valid_mae(args, model, valid_loader, device)
            test_loss = valid_mae(args, model, test_loader, device)

            valid_recon_loss = valid_mae_reconstruction(args, model, valid_loader, device)
            test_recon_loss = valid_mae_reconstruction(args, model, test_loader, device)

            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}, Test Loss: {test_loss:.7f}")
            print(f"          Full Recon - Valid: {valid_recon_loss:.7f}, Test: {test_recon_loss:.7f}")

            early_stopping(valid_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        # Test
        print(f">>>>>>>start testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        model.eval()

        test_losses_masked = []
        test_losses_recon = []
        criterion = nn.MSELoss()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)

                loss, pred, mask = model(batch_x)
                test_losses_masked.append(loss.item())

                pred_full = model.reconstruct(batch_x)
                loss_recon = criterion(pred_full, batch_x)
                test_losses_recon.append(loss_recon.item())

        final_test_loss_masked = np.average(test_losses_masked)
        final_test_loss_recon = np.average(test_losses_recon)

        print(f"Final Test Masked Recon Loss (MSE): {final_test_loss_masked:.7f}")
        print(f"Final Test Full Recon Loss (MSE): {final_test_loss_recon:.7f}")

        # 保存结果
        result_path = './result/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open("result_mae.txt", 'a') as f:
            f.write(setting + "\n")
            f.write(f"Masked Recon MSE: {final_test_loss_masked:.7f}\n")
            f.write(f"Full Recon MSE: {final_test_loss_recon:.7f}\n")
            f.write('\n')

        # Save latents
        print(f"Saving latent representations...")
        model.eval()
        latents = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(device)
                latent = model.encode(batch_x)
                latents.append(latent.cpu().numpy())

        latents = np.concatenate(latents, axis=0)
        np.save(result_path + 'train_latents.npy', latents)
        print(f"Saved train latents shape: {latents.shape}")

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()