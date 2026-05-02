import os
import torch
import torch.nn as nn
import torch.backends
import numpy as np

import time
from torch import optim
import wandb

from utils.tools import EarlyStopping, adjust_learning_rate, visual

from my_utils import args_train, set_seed, acquire_device, get_data
from RevIN import RevIN


class AutoEncoder(nn.Module):
    """
    MLP AutoEncoder: 2-layer encoder + 2-layer decoder.
    Encodes/decodes along the last dimension (feature dim).
    Input: (batch, seq_len, enc_in); output: (batch, seq_len, enc_in).
    Latent: (batch, seq_len, d_model).
    """
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: 2 layers on last dimension
        self.encoder = nn.Sequential(
            nn.Linear(self.enc_in, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
        )

        # Decoder: 2 layers on last dimension
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.enc_in),
        )

    def encode(self, x):
        latent = self.encoder(x)
        return latent

    def decode(self, latent):
        x = self.decoder(latent)
        return x

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output

class AutoEncoder_Revin(nn.Module):
    """
    MLP AutoEncoder with RevIN: 2-layer encoder + 2-layer decoder + RevIN.
    Encodes/decodes along the last (feature) dimension.
    Input: (batch, seq_len, enc_in); output: (batch, seq_len, enc_in).
    Latent: (batch, seq_len, d_model).
    """
    def __init__(self, args):
        super(AutoEncoder_Revin, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # RevIN layer
        self.revin_affine = getattr(args, 'revin_affine', 1) == 1
        self.revin = RevIN(self.enc_in, affine=self.revin_affine)

        # Encoder: 2 layers on last dimension
        self.encoder = nn.Sequential(
            nn.Linear(self.enc_in, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
        )

        # Decoder: 2 layers on last dimension
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.enc_in),
        )

    def encode(self, x):
        x = self.revin(x, 'norm')
        latent = self.encoder(x)  # (batch, seq_len, d_model)
        return latent

    def decode(self, latent):
        # latent: (batch, seq_len, d_model)
        x = self.decoder(latent)  # (batch, seq_len, enc_in)
        x = self.revin(x, 'denorm')  # RevIN denormalization
        return x

    def forward(self, x):
        # x: (batch, seq_len, enc_in)
        latent = self.encode(x)
        output = self.decode(latent)
        return output


class TemporalAutoEncoder(nn.Module):
    """
    Temporal AutoEncoder: models the seq_len dimension (channel independence).
    Aligns with DLinear, PatchTST, etc.

    Input: (batch, seq_len, enc_in).
    Flow: permute -> (batch, enc_in, seq_len) -> encode -> (batch, enc_in, d_model)
          -> permute -> (batch, d_model, enc_in).
    Output: (batch, seq_len, enc_in).
    Latent: (batch, d_model, enc_in); enc_in kept last for TSF compatibility.
    TSF sees: seq_len = d_model, feature dim = enc_in (unchanged).
    """
    def __init__(self, args):
        super(TemporalAutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: 对每个变量的时间序列进行编码
        # (batch, enc_in, seq_len) → (batch, enc_in, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(self.seq_len, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
        )

        # Decoder: (batch, enc_in, d_model) -> (batch, enc_in, seq_len)
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
    Temporal 1D-CNN AutoEncoder: 对 seq_len 维度进行卷积建模 (Channel Independence)

    输入: (batch, seq_len, enc_in)
    输出: (batch, seq_len, enc_in)
    latent: (batch, d_model, enc_in) - 保持 enc_in 在最后一维
    """
    def __init__(self, args):
        super(TemporalConv1dAutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: MLP on time dimension
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


class Conv1dAutoEncoder(nn.Module):
    """
    1D-CNN AutoEncoder: 使用1D卷积沿时间维度进行编码
    输入: (batch, seq_len, enc_in)
    输出: (batch, seq_len, enc_in)
    latent: (batch, seq_len, d_model)
    """
    def __init__(self, args):
        super(Conv1dAutoEncoder, self).__init__()
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.d_ff = args.d_ff

        # Encoder: (batch, enc_in, seq_len) -> (batch, d_model, seq_len)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.enc_in, self.d_ff, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_ff),
            nn.ReLU(),
            nn.Conv1d(self.d_ff, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
        )

        # Decoder: 1D ConvTranspose
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.d_model, self.d_ff, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_ff),
            nn.ReLU(),
            nn.ConvTranspose1d(self.d_ff, self.enc_in, kernel_size=3, padding=1),
        )

    def encode(self, x):
        # x: (batch, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch, enc_in, seq_len)
        latent = self.encoder(x)  # (batch, d_model, seq_len)
        latent = latent.permute(0, 2, 1)  # (batch, seq_len, d_model)
        return latent

    def decode(self, latent):
        # latent: (batch, seq_len, d_model)
        latent = latent.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.decoder(latent)  # (batch, enc_in, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, enc_in)
        return x

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output


def get_autoencoder(args):
    """
    Select AutoEncoder architecture by ae_type.

    Args:
        args: must include ae_type:
            - 'MLP': MLP AE on enc_in dim (default)
            - 'MLP_REVIN': MLP AE with RevIN on enc_in dim
            - 'CNN': 1D-CNN AE on enc_in dim
            - 'Temporal': Temporal AE on seq_len dim (channel independence)
            - 'TemporalCNN': Temporal Conv1d AE on seq_len dim

    Returns:
        AutoEncoder model.

    Latent shapes:
        - MLP/MLP_REVIN/CNN: (batch, seq_len, d_model)
        - Temporal/TemporalCNN: (batch, d_model, enc_in)
    """
    ae_type = getattr(args, 'ae_type', 'MLP').upper()

    if ae_type == 'MLP':
        print(f"Using MLP AutoEncoder (enc_in → d_model)")
        print(f"  Latent shape: (batch, seq_len, d_model)")
        return AutoEncoder(args)
    elif ae_type == 'MLP_REVIN':
        print(f"Using MLP AutoEncoder with RevIN (enc_in → d_model)")
        print(f"  Latent shape: (batch, seq_len, d_model)")
        return AutoEncoder_Revin(args)
    elif ae_type == 'CNN':
        print(f"Using Conv1d AutoEncoder (enc_in → d_model)")
        print(f"  Latent shape: (batch, seq_len, d_model)")
        return Conv1dAutoEncoder(args)
    elif ae_type == 'TEMPORAL':
        print(f"Using Temporal AutoEncoder (seq_len → d_model, Channel Independence)")
        print(f"  Latent shape: (batch, d_model, enc_in)")
        return TemporalAutoEncoder(args)
    elif ae_type == 'TEMPORALCNN':
        print(f"Using Temporal Conv1d AutoEncoder (seq_len → d_model, Channel Independence)")
        print(f"  Latent shape: (batch, d_model, enc_in)")
        return TemporalConv1dAutoEncoder(args)
    else:
        raise ValueError(f"Unknown ae_type: {ae_type}. Choose from ['MLP', 'MLP_REVIN', 'CNN', 'Temporal', 'TemporalCNN']")


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

    ae_type = getattr(args, 'ae_type', 'MLP').upper()

    for iter in range(args.itr):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<")

        set_seed(args.seed)

        ae_loss_type = getattr(args, 'ae_loss', 'MSE').upper()
        use_lradj_value = getattr(args, 'use_lradj', 1)
        revin_affine_value = getattr(args, 'revin_affine', 1)

        if ae_type == 'MLP_REVIN':
            setting = f"AutoEncoder_{ae_type}_revin{revin_affine_value}_{ae_loss_type}_{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_dm{args.d_model}_dff{args.d_ff}_lradj{use_lradj_value}_{args.des}_{iter}"
        else:
            setting = f"AutoEncoder_{ae_type}_{ae_loss_type}_{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_dm{args.d_model}_dff{args.d_ff}_lradj{use_lradj_value}_{args.des}_{iter}"

        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # Initialize wandb
        # Use environment variable WANDB_PROJECT if set, otherwise use default
        wandb_project = os.environ.get('WANDB_PROJECT', 'LatentTSF-AutoEncoder')
        wandb.init(
            project=wandb_project,
            name=setting,
            config={
                "ae_type": ae_type,
                "ae_loss": ae_loss_type,
                "model_id": args.model_id,
                "dataset": args.data_path.split('.')[0],
                "features": args.features,
                "seq_len": args.seq_len,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "enc_in": args.enc_in,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "train_epochs": args.train_epochs,
                "patience": args.patience,
                "seed": args.seed,
                "iteration": iter,
                "use_lradj": use_lradj_value,
                "revin_affine": revin_affine_value if ae_type == 'MLP_REVIN' else 'N/A',
                "description": args.des
            },
            reinit=True
        )

        device = acquire_device(args)

        model = get_autoencoder(args).float()
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
        ae_loss_type = getattr(args, 'ae_loss', 'MSE').upper()
        if ae_loss_type == 'MAE':
            criterion = nn.L1Loss()
            print(f"Using MAE (L1Loss) for AutoEncoder training")
        else:
            criterion = nn.MSELoss()
            print(f"Using MSE (MSELoss) for AutoEncoder training")

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

                # Reconstruct input
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

            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "learning_rate": model_optim.param_groups[0]['lr']
            })

            early_stopping(valid_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rate if enabled
            use_lradj = getattr(args, 'use_lradj', 1)
            if use_lradj:
                adjust_learning_rate(model_optim, epoch+1, args)

        # load the best model
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        # Test: final reconstruction error
        print(f">>>>>>>start testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        model.eval()
        mae_criterion=nn.L1Loss()
        mse_criterion=nn.MSELoss()
        test_losses = []
        test_mse_losses = []
        test_mae_losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                mae_loss=mae_criterion(outputs, batch_x)
                mse_loss=mse_criterion(outputs, batch_x)
                test_losses.append(loss.item())
                test_mae_losses.append(mae_loss.item())
                test_mse_losses.append(mse_loss.item())

        final_test_loss = np.average(test_losses)
        final_test_mae_loss = np.average(test_mae_losses)
        final_test_mse_loss = np.average(test_mse_losses)
        print(f"Final Test Reconstruction Loss (MSE): {final_test_mse_loss:.7f}")
        print(f"Final Test Reconstruction Loss (MAE): {final_test_mae_loss:.7f}")

        # Log final test metrics to wandb
        wandb.log({
            "final_test_mse": final_test_mse_loss,
            "final_test_mae": final_test_mae_loss
        })

        # 保存结果
        result_path = './result/' + setting + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open("result_autoencoder.txt", 'a') as f:
            f.write(setting + "\n")
            f.write(f"Test Reconstruction MSE: {final_test_mse_loss:.7f}\n")
            f.write(f"Test Reconstruction MAE: {final_test_mae_loss:.7f}\n")
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
        # np.save(result_path + 'train_latents.npy', latents)
        # print(f"Saved train latents shape: {latents.shape}")

        # Log latent shape to wandb
        wandb.log({
            "latent_shape_batch": latents.shape[0],
            "latent_shape_dim1": latents.shape[1],
            "latent_shape_dim2": latents.shape[2]
        })

        # Finish wandb run
        wandb.finish()

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
