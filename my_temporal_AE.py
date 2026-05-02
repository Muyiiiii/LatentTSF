import os
import torch
import torch.nn as nn
import torch.backends
import numpy as np

import time
from torch import optim

from utils.tools import EarlyStopping, adjust_learning_rate, visual

from my_utils import args_train, set_seed, acquire_device, get_data


class TemporalAutoEncoder(nn.Module):
    """
    Temporal AutoEncoder: models seq_len dimension (channel independence).
    Aligns with DLinear, PatchTST, etc.

    Input: (batch, seq_len, enc_in).
    Flow: permute -> (batch, enc_in, seq_len) -> encode -> (batch, enc_in, d_model)
          -> permute -> (batch, d_model, enc_in).
    Output: (batch, seq_len, enc_in). Latent: (batch, d_model, enc_in).
    TSF sees: seq_len = d_model, feature dim = enc_in (unchanged).
    """
    def __init__(self, args):
        super(TemporalAutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: (batch, enc_in, seq_len) -> (batch, enc_in, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(self.seq_len, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
        )

        # Decoder: 从 latent 恢复时间序列
        # (batch, enc_in, d_model) → (batch, enc_in, seq_len)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.seq_len),
        )

    def encode(self, x):
        # x: (batch, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch, enc_in, seq_len)
        latent = self.encoder(x)  # (batch, enc_in, d_model)
        latent = latent.permute(0, 2, 1)  # (batch, d_model, enc_in)
        return latent

    def decode(self, latent):
        # latent: (batch, d_model, enc_in)
        latent = latent.permute(0, 2, 1)  # (batch, enc_in, d_model)
        x = self.decoder(latent)  # (batch, enc_in, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, enc_in)
        return x

    def forward(self, x):
        # x: (batch, seq_len, enc_in)
        latent = self.encode(x)  # (batch, d_model, enc_in)
        output = self.decode(latent)  # (batch, seq_len, enc_in)
        return output


class TemporalConv1dAutoEncoder(nn.Module):
    """
    Temporal 1D-CNN AutoEncoder: convolves along seq_len (channel independence).
    Input: (batch, seq_len, enc_in); output: (batch, seq_len, enc_in).
    Latent: (batch, d_model, enc_in).
    """
    def __init__(self, args):
        super(TemporalConv1dAutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: MLP 对时间维度编码
        self.encoder = nn.Sequential(
            nn.Linear(self.seq_len, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.seq_len),
        )

    def encode(self, x):
        # x: (batch, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch, enc_in, seq_len)
        latent = self.encoder(x)  # (batch, enc_in, d_model)
        latent = latent.permute(0, 2, 1)  # (batch, d_model, enc_in)
        return latent

    def decode(self, latent):
        # latent: (batch, d_model, enc_in)
        latent = latent.permute(0, 2, 1)  # (batch, enc_in, d_model)
        x = self.decoder(latent)  # (batch, enc_in, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, enc_in)
        return x

    def forward(self, x):
        latent = self.encode(x)  # (batch, d_model, enc_in)
        output = self.decode(latent)  # (batch, seq_len, enc_in)
        return output


def get_temporal_autoencoder(args):
    """
    Select Temporal AutoEncoder by ae_type.

    Args:
        args: must include ae_type:
            - 'Temporal': MLP on seq_len dim
            - 'TemporalCNN': Conv1d on seq_len dim

    Returns:
        Temporal AutoEncoder. Latent shape: (batch, d_model, enc_in).
    """
    ae_type = getattr(args, 'ae_type', 'Temporal')

    if ae_type == 'Temporal':
        print(f"Using Temporal AutoEncoder (seq_len → d_model, Channel Independence)")
        print(f"  Latent shape: (batch, d_model, enc_in)")
        return TemporalAutoEncoder(args)
    elif ae_type == 'TemporalCNN':
        print(f"Using Temporal Conv1d AutoEncoder (seq_len → d_model, Channel Independence)")
        print(f"  Latent shape: (batch, d_model, enc_in)")
        return TemporalConv1dAutoEncoder(args)
    else:
        raise ValueError(f"Unknown ae_type: {ae_type}. Choose from ['Temporal', 'TemporalCNN']")


def valid_autoencoder(args, model, valid_loader, criterion, device):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)

            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


if __name__ == "__main__":
    args = args_train()

    ae_type = getattr(args, 'ae_type', 'Temporal')

    for iter in range(args.itr):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<")

        set_seed(args.seed)

        setting = f"AETemporal_{ae_type}_{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_dm{args.d_model}_dff{args.d_ff}_{args.des}_{iter}"

        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")

        device = acquire_device(args)

        model = get_temporal_autoencoder(args).float()
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

        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

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

                # AutoEncoder: 重建输入
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_x)
                        train_loss.append(loss.item())
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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

            print(f"Epoch: {epoch+1}, cost time: {time.time()-epoch_time:.3f}s")
            train_loss = np.average(train_loss)
            valid_loss = valid_autoencoder(args, model, valid_loader, criterion, device)
            test_loss = valid_autoencoder(args, model, test_loader, criterion, device)

            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}, Test Loss: {test_loss:.7f}")

            early_stopping(valid_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, args)

        # load the best model
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        # Test: final reconstruction error
        print(f">>>>>>>start testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        model.eval()
        test_losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                test_losses.append(loss.item())

        final_test_loss = np.average(test_losses)
        print(f"Final Test Reconstruction Loss (MSE): {final_test_loss:.7f}")

        # 保存结果
        result_path = './result/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open("result_temporal_autoencoder.txt", 'a') as f:
            f.write(setting + "\n")
            f.write(f"Test Reconstruction MSE: {final_test_loss:.7f}\n")
            f.write('\n')

        # Save latent representations
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
