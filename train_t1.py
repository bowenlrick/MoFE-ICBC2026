import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Info] Random seed set to {seed}")

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'window_size': 32,
    'forecast_horizon': 1,
    'smooth_window': 1,

    'd_model': 48,

    'modes': 8,
    'num_experts': 2,
    'dropout': 0.5,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 0.0005,
    'weight_decay': 1e-2,
    'loss_alpha': 0.20, 
    'loss_beta': 0.50, 
    'l1_lambda': 1e-06,
    'moe_smooth_lambda': 1e-03,

    'ortho_lambda': 0.05, 

    'patience': 20,
    'min_delta': 1e-5,

    'num_blocks': 1,
    'sparsity_threshold': 0.001,

    'hard_thresholding_fraction': 0.75,

    'hidden_size_factor': 1
}

def load_and_process_data(config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, './data/train.csv')
    val_path = os.path.join(base_dir, './data/val.csv')
    test_path = os.path.join(base_dir, './data/test.csv')

    print(f"[Info] Loading data from:\n      {train_path}\n      {val_path}\n      {test_path}")

    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find the dataset files.\nOriginal error: {e}")

    def calculate_indicators(df):
        prices = df['Close'].values.astype(float)
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_norm = (rsi / 50.0) - 1.0

        ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        macd_hist = macd_line - signal_line
        macd_norm = macd_hist / (np.std(macd_hist) + 1e-9)
        return rsi_norm, macd_norm

    def process_dataset(df, config):
        if 'Timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        else:
            df['Date'] = pd.to_datetime(df.get('Date', pd.date_range(start='2018-01-01', periods=len(df))))
        df = df.sort_values('Date').reset_index(drop=True)

        try:
            p_open = df['Open'].values.astype(float)
            p_high = df['High'].values.astype(float)
            p_low = df['Low'].values.astype(float)
            p_close = df['Close'].values.astype(float)
            p_volume = df['Volume'].values.astype(float)
        except KeyError:
            p_close = df['Close'].values.astype(float)
            p_open = p_high = p_low = p_close
            p_volume = np.ones_like(p_close)

        rsi, macd = calculate_indicators(df)

        def get_log_diff(series):
            series = np.maximum(series, 1e-8)
            return np.diff(np.log(series))

        ret_open = get_log_diff(p_open)
        ret_high = get_log_diff(p_high)
        ret_low = get_log_diff(p_low)
        ret_close = get_log_diff(p_close)

        vol_series = pd.Series(p_volume)
        vol_ma = vol_series.rolling(window=20, min_periods=1).mean().values
        feature_rvol = np.log((p_volume / (vol_ma + 1e-9)) + 1e-9)
        feature_rvol = feature_rvol[1:]

        rsi = rsi[1:]
        macd = macd[1:]
        log_ret_dates = df['Date'].values[1:]
        vol_proxy = pd.Series(ret_close).rolling(window=5).std().fillna(0).values

        smooth_window = config['smooth_window']
        target_returns_smooth = pd.Series(ret_close).rolling(window=smooth_window).mean().fillna(0).values

        window_size = config['window_size']
        n_out = config['forecast_horizon']

        X_list, y_list, dates_list, base_prices_list = [], [], [], []
        num_samples = len(ret_close) - window_size - n_out + 1

        for i in range(num_samples):
            raw_features = np.stack([
                ret_open[i: i + window_size],
                ret_high[i: i + window_size],
                ret_low[i: i + window_size],
                ret_close[i: i + window_size],
                vol_proxy[i: i + window_size],
                feature_rvol[i: i + window_size],
                rsi[i: i + window_size],
                macd[i: i + window_size]
            ], axis=-1)

            w_mean = np.mean(raw_features, axis=0, keepdims=True)
            w_std = np.std(raw_features, axis=0, keepdims=True) + 1e-9
            norm_features = (raw_features - w_mean) / w_std

            X_list.append(norm_features)
            y_list.append(target_returns_smooth[i + window_size: i + window_size + n_out])
            dates_list.append(log_ret_dates[i + window_size + n_out - 1])
            base_prices_list.append(p_close[i + window_size - 1])

        if len(X_list) == 0: return np.array([]), np.array([]), np.array([]), np.array([])
        return np.array(X_list), np.array(y_list), np.array(dates_list), np.array(base_prices_list)

    X_train_raw, y_train_raw, dates_train, base_prices_train = process_dataset(df_train, config)
    X_val_raw, y_val_raw, dates_val, base_prices_val = process_dataset(df_val, config)
    X_test_raw, y_test_raw, dates_test, base_prices_test = process_dataset(df_test, config)

    scaler_y_params = {}
    if len(y_train_raw) > 0:
        scaler_y_params['mean'] = np.mean(y_train_raw)
        scaler_y_params['scale'] = np.std(y_train_raw) + 1e-9

    class SimpleScaler:
        def __init__(self, params): self.y_params = params

    scaler = SimpleScaler(scaler_y_params)

    def transform_data(X_raw, y_raw, is_training=False):
        if len(X_raw) == 0: return None, None
        X = torch.FloatTensor(X_raw).to(config['device'])
        if is_training:
            y_raw = np.clip(y_raw, -0.05, 0.05)
        y = (y_raw - scaler_y_params['mean']) / scaler_y_params['scale']
        return X, torch.FloatTensor(y).to(config['device'])

    X_train, y_train = transform_data(X_train_raw, y_train_raw, is_training=True)
    X_val, y_val = transform_data(X_val_raw, y_val_raw, is_training=False)
    X_test, y_test = transform_data(X_test_raw, y_test_raw, is_training=False)

    return {
        'train': (X_train, y_train, dates_train, base_prices_train),
        'test': (X_test, y_test, dates_test, base_prices_test),
        'val': (X_val, y_val, dates_val, base_prices_val)
    }, scaler

class AFNO1D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
                self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
                torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)

        return x + bias


class FNOExpert(nn.Module):
    def __init__(self, modes, d_model, out_channels, num_blocks, sparsity_threshold, hard_thresholding_fraction,
                 dropout_p=0.2):
        super().__init__()
        self.conv1 = AFNO1D(
            hidden_size=d_model,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction
        )
        self.w0 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm1d(d_model)
        )
        self.w1 = nn.Conv1d(d_model, out_channels, 1)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x_afno = self.conv1(x)
        x_linear = x.permute(0, 2, 1)
        x_linear = self.w0(x_linear)
        x_linear = x_linear.permute(0, 2, 1)
        x = self.dropout(self.act(x_afno + x_linear))
        x = x.permute(0, 2, 1)
        return (self.w1(x))[:, :, -1]


class MoE_GibbsFNO(nn.Module):
    def __init__(self, num_experts, modes, d_model, window_size, out_channels,
                 num_blocks, sparsity_threshold, hard_thresholding_fraction, dropout_p=0.2):
        super().__init__()
        self.experts = nn.ModuleList([
            FNOExpert(modes, d_model, out_channels, num_blocks, sparsity_threshold, hard_thresholding_fraction,
                      dropout_p)
            for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(nn.Linear(window_size * d_model, 32), nn.LeakyReLU(0.1), nn.Dropout(dropout_p),
                                    nn.Linear(32, num_experts), nn.Softmax(dim=1))

    def forward(self, x):
        weights = self.gating(x.reshape(x.shape[0], -1))  
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        moe_out = torch.bmm(weights.unsqueeze(1), expert_outputs).squeeze(1)  
        return moe_out, weights


class HybridDirectionalLoss(nn.Module):
    def __init__(self, alpha=0.35, beta=0.40):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        weights = (torch.log1p(torch.abs(target) * 100) + 1.0)
        weights = weights / weights.mean()
        loss_mse = (self.mse(pred, target) * weights).mean()
        pf, tf, wf = pred.reshape(-1), target.reshape(-1), weights.reshape(-1)
        is_correct_dir = (torch.sign(pf) == torch.sign(tf)).float()
        amplitude_error = torch.clamp(torch.abs(tf) - torch.abs(pf), min=0)
        is_big_move = (torch.abs(tf) > 0.01).float()
        direction_error = (1.0 - is_correct_dir)
        be_penalty = 1.0 + (is_big_move * (direction_error * 3.0 + amplitude_error * 50.0))
        loss_sign = (self.bce(pf * 10.0, (tf > 0).float()) * wf * be_penalty).mean()
        vx, vy = pf - pf.mean(), tf - tf.mean()
        loss_ic = 1.0 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        return self.alpha * loss_mse + (1 - self.alpha - self.beta) * loss_ic + self.beta * loss_sign


class BitCoinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, forecast_horizon,
                 dropout=0.2, modes=16, num_experts=4,
                 num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
        super(BitCoinPredictor, self).__init__()
        self.lifting = nn.Linear(input_dim, hidden_dim)  # Stem Block
        self.moe_fno = MoE_GibbsFNO(
            num_experts, modes, hidden_dim, window_size, hidden_dim,
            num_blocks, sparsity_threshold, hard_thresholding_fraction, dropout
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_horizon)
        )

    def forward(self, x):
        x = self.lifting(x) 
        x, weights = self.moe_fno(x)  
        out = self.projection(x)  
        return out, weights

def train_model(model, datasets, config, scaler):
    X_train, y_train, _, _ = datasets['train']
    X_val, y_val, _, _ = datasets['val']

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = HybridDirectionalLoss(alpha=config['loss_alpha'], beta=config['loss_beta'])

    best_val_loss = float('inf')
    early_stop_count = 0
    best_model_state = None

    print(f"\nTraining T+{config['forecast_horizon']} with HybridLoss & Optimized AFNO...")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train),
                                               batch_size=config['batch_size'], shuffle=True)

    for epoch in range(config['epochs']):
        model.train()
        total_loss_val = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred, weights = model(bx)
            base_loss = criterion(pred, by)
            l1_reg = config['l1_lambda'] * sum(p.abs().sum() for p in model.parameters())
            smooth_reg = config['moe_smooth_lambda'] * torch.mean((weights[1:] - weights[:-1]) ** 2)

            w_norm = weights / (weights.norm(dim=0, keepdim=True) + 1e-9)
            corr_matrix = torch.mm(w_norm.t(), w_norm)
            identity = torch.eye(corr_matrix.size(0), device=config['device'])
            ortho_loss = config.get('ortho_lambda', 0.0) * torch.norm(corr_matrix - identity)

            total_loss = base_loss + l1_reg + smooth_reg + ortho_loss
            total_loss.backward()
            optimizer.step()
            total_loss_val += total_loss.item()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            vp_np = val_pred.cpu().numpy().flatten()
            vy_np = y_val.cpu().numpy().flatten()
            val_ic = np.corrcoef(vp_np, vy_np)[0, 1] if len(vp_np) > 1 else 0

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch:03d}] | Total Loss: {total_loss_val / len(train_loader):.6f} | Val Loss: {val_loss:.6f} | Val IC: {val_ic:.4f}")

        if val_loss < (best_val_loss - config['min_delta']):
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= config['patience']:
                print(f"[Info] Early stopping at epoch {epoch}. Restoring best weights...")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def get_feature_importance(model, X):
    model.eval()
    X_grad = X.clone().detach().requires_grad_(True)
    pred, _ = model(X_grad)
    pred.sum().backward()
    grads = X_grad.grad.abs()
    feature_importance = grads.mean(dim=1)
    feature_importance = feature_importance / (feature_importance.sum(dim=1, keepdim=True) + 1e-9)
    return feature_importance.cpu().numpy()

def evaluate_market_value(model, X, y, dates, base_prices, scaler, config):
    model.eval()
    with torch.no_grad():
        pred_scaled, _ = model(X)

    pred_scaled = pred_scaled.cpu().numpy().flatten()
    y_true_scaled = y.cpu().numpy().flatten()

    mean_y = scaler.y_params['mean']
    scale_y = scaler.y_params['scale']

    pred_log_ret = pred_scaled * scale_y + mean_y
    actual_log_ret = y_true_scaled * scale_y + mean_y

    pred_prices = base_prices * np.exp(pred_log_ret)
    actual_prices = base_prices * np.exp(actual_log_ret)

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)

    r2 = r2_score(actual_prices, pred_prices)
    ic, _ = pearsonr(pred_log_ret, actual_log_ret)

    correct_dir = (np.sign(pred_log_ret) == np.sign(actual_log_ret))
    win_rate = np.mean(correct_dir)

    pos_actual = actual_log_ret[correct_dir & (actual_log_ret > 0)]
    neg_actual = actual_log_ret[correct_dir & (actual_log_ret < 0)]
    pl_ratio = (np.mean(pos_actual) / abs(np.mean(neg_actual))) if (len(neg_actual) > 0 and len(pos_actual) > 0) else 0

    threshold = 0.002
    strat_returns = np.where(np.abs(pred_log_ret) > threshold,
                             np.sign(pred_log_ret) * actual_log_ret,
                             0)

    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns)
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(365.0)

    cum_strat_log_ret = np.cumsum(strat_returns)
    total_net_value = np.exp(cum_strat_log_ret[-1])

    return {
        'rmse': rmse, 'mae': mae, 'r2': r2, 'ic': ic, 
        'da': win_rate, 'pl_ratio': pl_ratio, 'sharpe': sharpe,
        'total_net_value': total_net_value,
        'pred_prices': pred_prices, 'actual_prices': actual_prices,
        'pred_log_ret': pred_log_ret, 'actual_log_ret': actual_log_ret
    }

def count_flops(model, input_shape, device):
    flops = 0.0
    handles = []

    def linear_hook(m, x, y):
        nonlocal flops
        total_input_elements = x[0].numel()
        batch_seq = total_input_elements / m.in_features

        layer_flops = 2 * m.in_features * m.out_features * batch_seq
        if m.bias is not None:
            layer_flops += m.out_features * batch_seq
        flops += layer_flops

    def conv1d_hook(m, x, y):
        nonlocal flops
        output = y
        batch_size = output.size(0)
        output_len = output.size(2)

        layer_flops = 2 * m.in_channels * m.kernel_size[0] * m.out_channels * output_len * batch_size
        if m.bias is not None:
            layer_flops += m.out_channels * output_len * batch_size
        flops += layer_flops

    def rnn_hook(m, x, y):
        nonlocal flops
        input_t = x[0]
        batch_size = input_t.size(0)
        seq_len = input_t.size(1)
        input_dim = input_t.size(2)

        hidden_dim = m.hidden_size
        num_layers = m.num_layers
        bi = 2 if m.bidirectional else 1

        if isinstance(m, nn.LSTM):
            gates = 4
        elif isinstance(m, nn.GRU):
            gates = 3
        else:
            gates = 1

        layer1_ops = 2 * gates * hidden_dim * (input_dim + hidden_dim) * bi

        other_layers_ops = 0
        if num_layers > 1:
            other_layers_ops = (num_layers - 1) * 2 * gates * hidden_dim * (hidden_dim * bi + hidden_dim) * bi

        total_ops = (layer1_ops + other_layers_ops) * batch_size * seq_len
        flops += total_ops

    def attn_hook(m, x, y):
        nonlocal flops
        q = x[0]
        batch_size = q.size(0)
        seq_len = q.size(1)
        embed_dim = m.embed_dim

        proj_ops = 3 * (2 * embed_dim * embed_dim) * seq_len * batch_size

        attn_ops = 2 * (seq_len ** 2) * embed_dim * batch_size

        out_ops = 2 * embed_dim * embed_dim * seq_len * batch_size

        flops += (proj_ops + attn_ops + out_ops)

    def afno_hook(m, x, y):
        nonlocal flops
        inp = x[0]
        B, N, C = inp.shape
        fft_ops = 2 * 5 * N * torch.log2(torch.tensor(N, dtype=torch.float).to(inp.device)) * C * B
        kept_modes = int((N // 2 + 1) * m.hard_thresholding_fraction)
        block_ops = 8 * B * kept_modes * m.num_blocks * (2 * m.block_size ** 2 * m.hidden_size_factor)
        flops += (fft_ops + block_ops)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            handles.append(m.register_forward_hook(rnn_hook))
        elif isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(attn_hook))
        elif m.__class__.__name__ == 'AFNO1D':
            handles.append(m.register_forward_hook(afno_hook))

    original_mode = model.training
    model.eval()
    with torch.no_grad():
        dummy_input = torch.ones(input_shape).to(device)
        try:
            model(dummy_input)
        except Exception as e:
            print(f"[Warning] FLOPs count failed: {e}")
            flops = 0

    model.train(original_mode)
    for h in handles:
        h.remove()

    return int(flops)

if __name__ == '__main__':
    setup_seed(42)

    datasets, scaler = load_and_process_data(CONFIG)

    X_train_sample = datasets['train'][0]
    input_dim = X_train_sample.shape[-1]
    print(f"[Info] Detected Input Dimension: {input_dim}")

    model = BitCoinPredictor(
        input_dim=input_dim,
        hidden_dim=CONFIG['d_model'],
        window_size=CONFIG['window_size'],
        forecast_horizon=CONFIG['forecast_horizon'],
        dropout=CONFIG['dropout'],
        modes=CONFIG['modes'],
        num_experts=CONFIG['num_experts'],
        num_blocks=CONFIG['num_blocks'],
        sparsity_threshold=CONFIG['sparsity_threshold'],
        hard_thresholding_fraction=CONFIG['hard_thresholding_fraction']
    ).to(CONFIG['device'])

    model = train_model(model, datasets, CONFIG, scaler)

    X_test_t, y_test_t, test_dates, base_prices_test = datasets['test']

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE (Hybrid Loss + Optimized AFNO)")
    print("=" * 70)

    m_test = evaluate_market_value(model, X_test_t, y_test_t, test_dates, base_prices_test, scaler, CONFIG)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops_input_shape = (1, CONFIG['window_size'], input_dim)
    total_flops = count_flops(model, flops_input_shape, CONFIG['device'])

    print(f"RMSE: {m_test['rmse']:.4f} | MAE: {m_test['mae']:.4f} | R2: {m_test['r2']:.4f}")
    print(f"IC:   {m_test['ic']:.4f} | DA: {m_test['da'] * 100:.2f}%")
    print(f"Win/Loss Ratio: {m_test['pl_ratio']:.2f}")
    print(f"Total Net Value: {m_test['total_net_value']:.2f}x (Cumulative ROI)")
    print(f"Sharpe Ratio:   {m_test['sharpe']:.2f}")
    print(f"Model Params:   {total_params:,} (Trainable: {trainable_params:,})")
    print(f"FLOPs:          {total_flops:,} (approx. per inference)")
    print("=" * 70)

    _, weights_t = model(X_test_t)
    weights_np = weights_t.detach().cpu().numpy()

    feature_imp_np = get_feature_importance(model, X_test_t)

    fig, axes = plt.subplots(5, 1, figsize=(12, 28))

    axes[0].plot(test_dates, m_test['actual_prices'], 'k-', label='Actual (T+N)', alpha=0.7)
    axes[0].plot(test_dates, m_test['pred_prices'], 'r--', label='Pred (T+N)')
    axes[0].set_title(f"Price Prediction (Horizon: {CONFIG['forecast_horizon']} Days)")
    axes[0].legend();
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(test_dates[-100:], m_test['actual_log_ret'][-100:], color='gray', alpha=0.3, label='Actual Cum Ret')
    axes[1].plot(test_dates[-100:], m_test['pred_log_ret'][-100:], 'b-o', markersize=3, label='Pred Cum Ret')
    axes[1].set_title(f"Cumulative Log Returns (T+{CONFIG['forecast_horizon']}) - Last 100 Steps")
    axes[1].legend();
    axes[1].grid(True, alpha=0.3)

    actual_ret, pred_ret = m_test['actual_log_ret'], m_test['pred_log_ret']
    axes[2].scatter(actual_ret, pred_ret, alpha=0.5, s=15, c='purple')
    min_v = min(actual_ret.min(), pred_ret.min())
    max_v = max(actual_ret.max(), pred_ret.max())
    axes[2].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.8)
    axes[2].set_title(f"Correlation Scatter (IC: {m_test['ic']:.4f})")
    axes[2].grid(True, alpha=0.3)

    axes[3].stackplot(test_dates, weights_np.T, alpha=0.8)
    axes[3].set_title("MoE Expert Activation Weights")
    axes[3].grid(True, alpha=0.2)

    feature_labels = ['Open', 'High', 'Low', 'Close', 'Vol', 'RVol', 'RSI', 'MACD']
    axes[4].stackplot(test_dates, feature_imp_np.T, labels=feature_labels, alpha=0.8)
    axes[4].set_title("AI Input Feature Attention (Gradient-based Importance)")
    axes[4].legend(loc='upper left', fontsize='small')
    axes[4].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('bitcoin_prediction_optimized_final.png')
    print("[Info] Visualization saved as 'bitcoin_prediction_optimized_final.png'")