import torch

# ------------------------
# WGAN losses (scalar critic)
# ------------------------

def wgan_d_loss(d_real, d_fake):
    # d_*: (B,)
    return d_fake.mean() - d_real.mean()


def wgan_g_loss(d_fake):
    # d_fake: (B,)
    return -d_fake.mean()


# ------------------------
# Gradient Penalty (WGAN-GP, scalar critic)
# ------------------------

def gradient_penalty_wgan_scalar(d_out, x):
    """
    d_out: (B,) scalar critic output
    x:     (B, T, C)
    """
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def tail_quantile_loss(x_fake, x_real, q=0.95, channel_weights=None):
    if isinstance(q, (list, tuple)):
        qs = [float(v) for v in q]
    else:
        qs = [float(q)]
    for v in qs:
        if v <= 0.0 or v >= 1.0:
            raise ValueError(f"q must be in (0,1), got {v}")
    fake_flat = x_fake.reshape(-1, x_fake.size(-1))
    real_flat = x_real.reshape(-1, x_real.size(-1))
    losses = []
    for v in qs:
        fake_q = torch.quantile(fake_flat, v, dim=0)
        real_q = torch.quantile(real_flat, v, dim=0)
        diff = torch.abs(fake_q - real_q)
        if channel_weights is not None:
            w = torch.as_tensor(channel_weights, dtype=diff.dtype, device=diff.device)
            if w.numel() != diff.numel():
                raise ValueError(
                    f"channel_weights length {w.numel()} must equal channels {diff.numel()}"
                )
            w = w / torch.clamp(w.mean(), min=1e-6)
            diff = diff * w
        losses.append(torch.mean(diff))
    return torch.mean(torch.stack(losses))


def tail_quantile_loss_channel(x_fake, x_real, q=0.95, channel=1):
    q = float(q)
    if q <= 0.0 or q >= 1.0:
        raise ValueError(f"q must be in (0,1), got {q}")
    xf = x_fake[..., channel].reshape(-1)
    xr = x_real[..., channel].reshape(-1)
    fake_q = torch.quantile(xf, q)
    real_q = torch.quantile(xr, q)
    return torch.abs(fake_q - real_q)


def mean_channel_loss(x_fake, x_real, channel=2):
    xf = x_fake[..., channel]
    xr = x_real[..., channel]
    return torch.abs(xf.mean() - xr.mean())


def channel_stats_loss(
    x_fake,
    x_real,
    channels=None,
    channel_weights=None,
    use_mean=True,
    use_std=True,
    include_ramp=True,
):
    if channels is None:
        channels = list(range(x_fake.size(-1)))
    if x_fake.size(1) < 2 or x_real.size(1) < 2:
        include_ramp = False
    df = x_fake[:, 1:] - x_fake[:, :-1]
    dr = x_real[:, 1:] - x_real[:, :-1]
    if include_ramp:
        df = torch.abs(df)
        dr = torch.abs(dr)
    losses = []
    for ch in channels:
        w = None
        if channel_weights is not None:
            w = float(channel_weights[ch])
        xf = x_fake[..., ch]
        xr = x_real[..., ch]
        if use_mean:
            loss = torch.abs(xf.mean() - xr.mean())
            losses.append(loss if w is None else loss * w)
        if use_std:
            loss = torch.abs(xf.std(unbiased=False) - xr.std(unbiased=False))
            losses.append(loss if w is None else loss * w)
        if include_ramp:
            rf = df[..., ch]
            rr = dr[..., ch]
            if use_mean:
                loss = torch.abs(rf.mean() - rr.mean())
                losses.append(loss if w is None else loss * w)
            if use_std:
                loss = torch.abs(rf.std(unbiased=False) - rr.std(unbiased=False))
                losses.append(loss if w is None else loss * w)
    if not losses:
        return torch.tensor(0.0, device=x_fake.device)
    return torch.mean(torch.stack(losses))


def rolling_stats_loss(x_fake, x_real, win=8, eps=1e-6):
    # Rolling mean/std over time dimension, averaged across channels.
    if win <= 1 or x_fake.size(1) < win or x_real.size(1) < win:
        return x_fake.sum() * 0.0
    # x_*: (B, T, C) -> (B, C, T)
    xf = x_fake.transpose(1, 2)
    xr = x_real.transpose(1, 2)
    k = int(win)
    # rolling mean via avg pool
    mf = torch.nn.functional.avg_pool1d(xf, kernel_size=k, stride=1)
    mr = torch.nn.functional.avg_pool1d(xr, kernel_size=k, stride=1)
    # rolling std: sqrt(E[x^2] - E[x]^2)
    mf2 = torch.nn.functional.avg_pool1d(xf * xf, kernel_size=k, stride=1)
    mr2 = torch.nn.functional.avg_pool1d(xr * xr, kernel_size=k, stride=1)
    sf = torch.sqrt(torch.clamp(mf2 - mf * mf, min=0.0) + eps)
    sr = torch.sqrt(torch.clamp(mr2 - mr * mr, min=0.0) + eps)
    return (mf - mr).abs().mean() + (sf - sr).abs().mean()


def acf_loss(x_fake, x_real, max_lag=6, eps=1e-6):
    # Autocorrelation loss for lags 1..max_lag, averaged across batch/channels.
    if max_lag <= 0 or x_fake.size(1) <= 1 or x_real.size(1) <= 1:
        return x_fake.sum() * 0.0
    max_lag = min(int(max_lag), x_fake.size(1) - 1, x_real.size(1) - 1)
    xf = x_fake - x_fake.mean(dim=1, keepdim=True)
    xr = x_real - x_real.mean(dim=1, keepdim=True)
    denom_f = xf.pow(2).mean(dim=1, keepdim=True) + eps
    denom_r = xr.pow(2).mean(dim=1, keepdim=True) + eps
    losses = []
    for lag in range(1, max_lag + 1):
        cf = (xf[:, :-lag] * xf[:, lag:]).mean(dim=1, keepdim=True) / denom_f
        cr = (xr[:, :-lag] * xr[:, lag:]).mean(dim=1, keepdim=True) / denom_r
        losses.append((cf - cr).abs().mean())
    return torch.mean(torch.stack(losses))


def corr_matrix_loss(x_fake, x_real, eps=1e-6):
    # Correlation matrix alignment loss to preserve cross-channel coupling.
    # x_*: (B, T, C) -> flatten (B*T, C)
    c = x_fake.size(-1)
    xf = x_fake.reshape(-1, c)
    xr = x_real.reshape(-1, c)
    xf = xf - xf.mean(dim=0, keepdim=True)
    xr = xr - xr.mean(dim=0, keepdim=True)
    cov_f = (xf.transpose(0, 1) @ xf) / max(xf.size(0) - 1, 1)
    cov_r = (xr.transpose(0, 1) @ xr) / max(xr.size(0) - 1, 1)
    std_f = torch.sqrt(torch.clamp(torch.diag(cov_f), min=eps))
    std_r = torch.sqrt(torch.clamp(torch.diag(cov_r), min=eps))
    corr_f = cov_f / (std_f[:, None] * std_f[None, :] + eps)
    corr_r = cov_r / (std_r[:, None] * std_r[None, :] + eps)
    return torch.mean(torch.abs(corr_f - corr_r))


def peak_event_loss(x_fake, x_real, channels=None, peak_temp=20.0):
    if channels is None:
        channels = list(range(x_fake.size(-1)))
    if not channels:
        return x_fake.sum() * 0.0
    t = torch.linspace(0.0, 1.0, x_fake.size(1), device=x_fake.device, dtype=x_fake.dtype)
    losses = []
    for ch in channels:
        xf = x_fake[..., ch]
        xr = x_real[..., ch]
        peak_val = torch.abs(xf.max(dim=1).values - xr.max(dim=1).values).mean()
        wf = torch.softmax(xf * float(peak_temp), dim=1)
        wr = torch.softmax(xr * float(peak_temp), dim=1)
        peak_pos_f = torch.sum(wf * t.unsqueeze(0), dim=1)
        peak_pos_r = torch.sum(wr * t.unsqueeze(0), dim=1)
        peak_pos = torch.abs(peak_pos_f - peak_pos_r).mean()
        losses.extend([peak_val, peak_pos])
    return torch.mean(torch.stack(losses))


def ramp_event_loss(x_fake, x_real, channels=None):
    if channels is None:
        channels = list(range(x_fake.size(-1)))
    if not channels or x_fake.size(1) < 2 or x_real.size(1) < 2:
        return x_fake.sum() * 0.0
    df = x_fake[:, 1:] - x_fake[:, :-1]
    dr = x_real[:, 1:] - x_real[:, :-1]
    losses = []
    for ch in channels:
        up_f = torch.relu(df[..., ch]).max(dim=1).values
        up_r = torch.relu(dr[..., ch]).max(dim=1).values
        dn_f = torch.relu(-df[..., ch]).max(dim=1).values
        dn_r = torch.relu(-dr[..., ch]).max(dim=1).values
        losses.append(torch.abs(up_f - up_r).mean())
        losses.append(torch.abs(dn_f - dn_r).mean())
    return torch.mean(torch.stack(losses))


def active_ratio_loss(x_fake, x_real, channels=None, threshold=0.05, sharpness=30.0):
    if channels is None:
        channels = list(range(x_fake.size(-1)))
    if not channels:
        return x_fake.sum() * 0.0
    losses = []
    thr = float(threshold)
    sharp = float(sharpness)
    for ch in channels:
        xf = x_fake[..., ch]
        xr = x_real[..., ch]
        af = torch.sigmoid((xf - thr) * sharp).mean(dim=1)
        ar = torch.sigmoid((xr - thr) * sharp).mean(dim=1)
        losses.append(torch.abs(af - ar).mean())
    return torch.mean(torch.stack(losses))
