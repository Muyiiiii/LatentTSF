import os
import csv
import argparse
import torch
import random
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
from utils.print_args import print_args
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.dtw_metric import dtw, accelerated_dtw
from utils.metrics import metric

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, CMoS, TimeBase, ModernTCN

model_dict = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'LightTS': LightTS,
    'Reformer': Reformer,
    'ETSformer': ETSformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'MICN': MICN,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'TiDE': TiDE,
    'FreTS': FreTS,
    'MambaSimple': MambaSimple,
    'TimeMixer': TimeMixer,
    'TSMixer': TSMixer,
    'SegRNN': SegRNN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    "SCINet": SCINet,
    'PAttn': PAttn,
    'TimeXer': TimeXer,
    'WPMixer': WPMixer,
    'MultiPatchFormer': MultiPatchFormer,
    'KANAD': KANAD,
    'MSGNet': MSGNet,
    'TimeFilter': TimeFilter,

    'CMoS': CMoS,
    'TimeBase': TimeBase,
    'ModernTCN': ModernTCN,

    # 'Sundial': Sundial,
    # 'TimeMoE': TimeMoE,
    # 'Chronos': Chronos,
    # 'Moirai': Moirai,
    # 'TiRex': TiRex,
    # 'TimesFM': TimesFM,
    # 'Chronos2': Chronos2
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Optional: enforce cuDNN determinism
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
def acquire_device(args):
    if args.use_gpu and args.gpu_type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    elif args.use_gpu and args.gpu_type == 'mps':
        device = torch.device('mps')
        print('Use GPU: mps')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def save_result_csv(args, setting, is_latent, mae, mse, rmse, mape, mspe, actual_epochs=None,
                    perceptual_weight=None, mse_weight=None, cosine_weight=None, reconstruction_weight=None,
                    record_seq_pred_len=False, max_memory_mb=None, training_time=None, use_lradj=None):
    """Save experiment results to a CSV file.

    Args:
        perceptual_weight, mse_weight, cosine_weight, reconstruction_weight:
            For latent mode pass the actual values; for original mode pass None.
        record_seq_pred_len: Whether to record seq_len, pred_len, step in CSV.
        max_memory_mb: Peak GPU memory in MB.
        training_time: Total training time in seconds.
        use_lradj: Whether learning rate adjustment was used (1=True, 0=False).
    """
    file_exists = os.path.exists(args.result_csv)
    align_weight = getattr(args, 'align_weight', None) if is_latent else None
    step = getattr(args, 'step', None)

    with open(args.result_csv, 'a', newline='') as f:
        writer = csv.writer(f)

        data_name = args.data_path.split('.')[0]

        if record_seq_pred_len:
            if not file_exists:
                writer.writerow(['setting', 'dataset', 'seq_len', 'pred_len', 'step', 'model', 'is_latent', 'align_weight',
                               'perceptual_weight', 'mse_weight', 'cosine_weight', 'reconstruction_weight',
                               'actual_epochs', 'use_lradj', 'mae', 'mse', 'rmse', 'mape', 'mspe', 'max_memory_mb', 'training_time'])
            writer.writerow([setting, data_name, args.seq_len, args.pred_len, step, args.model, is_latent, align_weight,
                            perceptual_weight, mse_weight, cosine_weight, reconstruction_weight,
                            actual_epochs, use_lradj, mae, mse, rmse, mape, mspe, max_memory_mb, training_time])
        else:
            if not file_exists:
                writer.writerow(['setting', 'dataset', 'model', 'is_latent', 'align_weight',
                               'perceptual_weight', 'mse_weight', 'cosine_weight', 'reconstruction_weight',
                               'actual_epochs', 'use_lradj', 'mae', 'mse', 'rmse', 'mape', 'mspe', 'max_memory_mb', 'training_time'])
            writer.writerow([setting, data_name, args.model, is_latent, align_weight,
                            perceptual_weight, mse_weight, cosine_weight, reconstruction_weight,
                            actual_epochs, use_lradj, mae, mse, rmse, mape, mspe, max_memory_mb, training_time])

    print(f"Results saved to {args.result_csv}")

def valid(args, model, valid_data, valid_loader, criterion, device):
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            pred=outputs.detach()
            true=batch_y.detach()
            loss = criterion(pred, true)
            
            total_loss.append(loss.item())
    total_loss = np.average(total_loss)
    model.train()
    return total_loss
    
def test(args, model, test_data, test_loader, setting, device, test=0):
    if test:
        print(f"loading model from {args.checkpoints + setting + '/' + 'checkpoint.pth'}")
        model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    preds=[]
    trues=[]
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
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            outputs= outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and args.inverse:
                shape=batch_y.shape
                if outputs.shape[-1] != batch_y.shape[-1]:
                    outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]
            
            pred=outputs
            true=batch_y
            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)
    print('test shape:', preds.shape, trues.shape)
    
    # result save
    folder_path = './result/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # dtw calculation
    if args.use_dtw:
        dtw_list = []
        manhattan_distance = lambda x, y: np.abs(x - y)
        for i in range(preds.shape[0]):
            x = preds[i].reshape(-1, 1)
            y = trues[i].reshape(-1, 1)
            if i % 100 == 0:
                print("calculating dtw iter:", i)
            d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
            dtw_list.append(d)
        dtw = np.array(dtw_list).mean()
    else:
        dtw = 'Not calculated'

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, dtw:{dtw}")
    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write(f"mae:{mae}, mse:{mse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, dtw:{dtw}")
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return mae, mse, rmse, mape, mspe

def args_train():
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--step', type=int, default=1, help='window sliding step for dataset')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--use_lradj', type=int, default=1, help='whether to use learning rate adjustment (1=True, 0=False)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=120, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # CMoS model
    parser.add_argument('--seg_size', type=int, default=24, help='segment size for CMoS')
    parser.add_argument('--num_map', type=int, default=4, help='number of mappings for CMoS')
    parser.add_argument('--cmos_kernel_size', type=int, default=4, help='conv kernel size for CMoS')
    parser.add_argument('--conv_stride', type=int, default=4, help='conv stride for CMoS')
    parser.add_argument('--use_pi', action='store_true', default=False, help='use period injection for CMoS')
    parser.add_argument('--period', type=int, default=24, help='period for CMoS period injection')

    # TimeBase model
    parser.add_argument('--period_len', type=int, default=24, help='period length for TimeBase')
    parser.add_argument('--basis_num', type=int, default=8, help='number of basis for TimeBase')
    parser.add_argument('--use_period_norm', action='store_true', default=False, help='use period normalization for TimeBase')
    parser.add_argument('--use_orthogonal', action='store_true', default=False, help='use orthogonal loss for TimeBase')

    # ModernTCN model
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio for ModernTCN')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample ratio for ModernTCN')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='FFN ratio for ModernTCN')
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[1, 1, 1, 1], help='number of blocks per stage for ModernTCN')
    parser.add_argument('--large_size', type=int, nargs='+', default=[31, 29, 27, 13], help='large kernel sizes for ModernTCN')
    parser.add_argument('--small_size', type=int, nargs='+', default=[5, 5, 5, 5], help='small kernel sizes for ModernTCN')
    parser.add_argument('--dims', type=int, nargs='+', default=[256, 256, 256, 256], help='dimensions for ModernTCN stages')
    parser.add_argument('--dw_dims', type=int, nargs='+', default=[256, 256, 256, 256], help='depthwise dimensions for ModernTCN')
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout for ModernTCN')
    parser.add_argument('--use_multi_scale', type=str2bool, default=False, help='use_multi_scale fusion')
    parser.add_argument('--revin', type=str2bool, default=True, help='use RevIN for ModernTCN')
    parser.add_argument('--affine', type=str2bool, default=True, help='affine for RevIN in ModernTCN')
    parser.add_argument('--subtract_last', type=str2bool, default=False, help='subtract last for RevIN in ModernTCN')
    parser.add_argument('--decomposition', type=str2bool, default=False, help='use decomposition for ModernTCN')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for decomposition in ModernTCN')
    parser.add_argument('--patch_size', type=int, default=90, help='patch size for ModernTCN')
    parser.add_argument('--patch_stride', type=int, default=16, help='patch stride for ModernTCN')

    # AutoEncoder for Latent TSF
    parser.add_argument('--use_latent', action='store_true', default=False,
                        help='Use latent space forecasting with AutoEncoder (v3 mode). If False, use original forecasting (v1 mode)')
    parser.add_argument('--autoencoder_path', type=str, default=None,
                        help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('--load_pretrained_ae', type=int, default=1,
                        help='Whether to load pretrained AE weights (0=False, 1=True)')
    parser.add_argument('--freeze_autoencoder', action='store_true', default=False,
                        help='Freeze autoencoder parameters during training')
    parser.add_argument('--align_weight', type=float, default=1.0,
                        help='Weight for embedding alignment loss (cosine similarity)')
    parser.add_argument('--use_residual', action='store_true', default=False,
                        help='Use residual connection: encoder embedding + model output')

    # V5: Encoder type selection (AE or MAE)
    parser.add_argument('--encoder_type', type=str, default='AE', choices=['AE', 'MAE'],
                        help='Encoder type: AE (AutoEncoder) or MAE (MaskedAutoEncoder)')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='Mask ratio for MAE training')

    # Result CSV
    parser.add_argument('--result_csv', type=str, default='result.csv',
                        help='CSV file path to save experiment results')
    parser.add_argument('--result_txt', type=str, default='result/result.txt',
                        help='TXT file path to append per-run test metrics (used by test_epoch)')
    parser.add_argument('--save_visual', action='store_true', default=False,
                        help='If set, save per-batch prediction-vs-truth PDFs into ./result/<setting>/ during testing')
    parser.add_argument('--visual_interval', type=int, default=20,
                        help='When --save_visual is set, save a PDF every N test batches')
    parser.add_argument('--load_checkpoint', type=int, default=1,
                        help='1 to reload best checkpoint inside test_epoch (default), 0 to evaluate the current in-memory model')

    # V10: Separate encoder/decoder freeze control and output decoder type
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder parameters during training')
    parser.add_argument('--freeze_decoder', action='store_true', default=False,
                        help='Freeze AE decoder parameters during training (only when output_decoder_type=AE)')
    parser.add_argument('--output_decoder_type', type=str, default='AE', choices=['AE', 'MLP'],
                        help='Output decoder type: AE (use AE decoder) or MLP (use new MLP decoder)')
    parser.add_argument('--encoder_lr', type=float, default=None,
                        help='Learning rate for encoder (if None, use learning_rate)')
    parser.add_argument('--decoder_lr', type=float, default=None,
                        help='Learning rate for decoder (if None, use learning_rate)')

    # Loss weights — defaults match the paper's main recipe (Eq. 5; Sec. 5.3.2):
    # L_total = mse_weight * ||Z_Y - Ẑ_Y||^2  +  cosine_weight * (1 - cos(Z_Y, Ẑ_Y))
    # mse_weight    = paper's α (Pred Weight)  = 10
    # cosine_weight = paper's β (Align Weight) = 15
    # perceptual_weight    : observation-space MSE on decoded output; OFF by default
    #                        ("we do not enable the perceptual loss by default", Sec. 5.3.1)
    # reconstruction_weight: extra L1 consistency D(Z_Y) vs Y; not part of paper recipe
    parser.add_argument('--mse_weight', type=float, default=10.0,
                        help="Weight on latent-space MSE term (paper's α, 'Pred Weight'); default 10")
    parser.add_argument('--cosine_weight', type=float, default=15.0,
                        help="Weight on latent-space cosine-alignment term (paper's β, 'Align Weight'); default 15")
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Weight on observation-space MSE of decoded forecast (L_Perc); disabled by default')
    parser.add_argument('--reconstruction_weight', type=float, default=0.0,
                        help='Weight on L1 reconstruction D(Z_Y) vs Y (not in paper); disabled by default')

    # AutoEncoder architecture type
    parser.add_argument('--ae_type', type=str, default='MLP', choices=['Temporal', 'TemporalCNN', 'MLP', 'MLP_REVIN', 'CNN'],
                        help='AutoEncoder architecture type: MLP or CNN (1D-Conv)')
    parser.add_argument('--ae_loss', type=str, default='MSE', choices=['MSE', 'MAE'],
                        help='AutoEncoder training loss function: MSE or MAE')
    parser.add_argument('--revin_affine', type=int, default=1,
                        help='RevIN affine parameter (1=True, 0=False)')

    # Latent LayerNorm
    parser.add_argument('--use_latent_norm', action='store_true', default=False,
                        help='Apply LayerNorm on latent space (z_x, z_pred)')

    # V13: MLP Adapters
    parser.add_argument('--use_mlp_adapter', type=int, default=1,
                        help='Use MLP adapters before/after TSF model (1=True, 0=False)')
    parser.add_argument('--mlp_hidden_mult', type=int, default=2,
                        help='MLP hidden dimension multiplier')

    # Gradient Accumulation
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')

    # Checkpoint Management
    parser.add_argument('--delete_checkpoint', action='store_true', default=False,
                        help='Delete model checkpoint after training and testing complete')

    # Embedding Saving
    parser.add_argument('--save_embedding', action='store_true', default=False,
                        help='Save embeddings every N epochs during training')
    parser.add_argument('--save_embedding_interval', type=int, default=2,
                        help='Interval (in epochs) for saving embeddings')

    # Experiment Label for WandB
    parser.add_argument('--exp_label', type=str, default=None,
                        help='Experiment label for WandB logging (e.g., "baseline", "exp1", "final")')

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    
    return args