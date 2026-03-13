from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_time_features


class MHSA(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        b, t, d = x.shape
        qkv = self.to_qkv(x).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D')
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class MHCBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.0, use_norm=True):
        super().__init__()
        self.h_pre = nn.Linear(dim, dim, bias=False)
        self.h_post = nn.Linear(dim, dim, bias=False)
        self.h_res = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.attn = MHSA(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # manifold-constrained hyper-connection
        # x_{l+1} = H_res x_l + H_post^T F(H_pre x_l, W_l)
        x_pre = self.h_pre(x)
        h = self.attn(self.norm1(x_pre))
        f = x_pre + self.dropout(h)
        y = F.gelu(self.fc1(self.norm2(f)))
        y = self.fc2(y)
        return self.h_res(x) + self.h_post(y)


class PhysicsMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        # x: (B, T, C), mask: (T, C) or (B, T, C)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        # Eq.(8): X_hat = X_tilde ⊙ m(w)
        return x * mask


class Generator(nn.Module):
    def __init__(
        self,
        seq_len,
        cond_dim,
        z_dim=64,
        model_dim=128,
        depth=4,
        heads=4,
        channels=4,
        dropout=0.1,
        use_film=True,
        use_channel_mixer=True,
        use_post_conv=True,
        use_baseline_residual=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.model_dim = model_dim
        self.z_proj = nn.Linear(z_dim, model_dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.time_proj = nn.Linear(2, model_dim)
        self.pre_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [MHCBlock(model_dim, heads=heads, dropout=dropout) for _ in range(depth)]
        )
        self.use_film = use_film
        self.use_baseline_residual = use_baseline_residual
        if use_film:
            self.film = nn.Linear(cond_dim, model_dim * 2)
        if use_baseline_residual:
            self.base_cond_proj = nn.Linear(cond_dim, model_dim)
            self.base_time_proj = nn.Linear(2, model_dim)
            self.base_mlp = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
            )
            self.base_to_out = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(channels)])
        self.channel_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(channels)
            ]
        )
        self.to_out = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(channels)])
        self.use_channel_mixer = use_channel_mixer
        if use_channel_mixer:
            self.channel_mixer = nn.Conv1d(channels, channels, kernel_size=1)
        self.use_post_conv = use_post_conv
        if use_post_conv:
            self.post_conv = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
        self.masker = PhysicsMask()
        self.register_buffer("time_feat", make_time_features(seq_len), persistent=False)

    def forward(self, z, cond, mask):
        # z: (B, z_dim), cond: (B, T, cond_dim) or (B, cond_dim)
        b = z.shape[0]
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        h = self.z_proj(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        c = self.cond_proj(cond)
        t = self.time_proj(self.time_feat).unsqueeze(0).to(h.dtype)
        h = h + c + t
        if self.use_film:
            film = self.film(cond)
            gamma, beta = film.chunk(2, dim=-1)
            h = h * (1 + gamma) + beta
        base = None
        if self.use_baseline_residual:
            hb = self.base_cond_proj(cond) + self.base_time_proj(self.time_feat).unsqueeze(0).to(h.dtype)
            hb = self.base_mlp(hb)
            base = torch.cat([self.base_to_out[i](hb) for i in range(len(self.base_to_out))], dim=-1)
        h = self.pre_conv(h.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        outs = [self.to_out[i](self.channel_mlps[i](h)) for i in range(len(self.to_out))]
        out = torch.cat(outs, dim=-1)
        if base is not None:
            out = out + base
        if self.use_channel_mixer:
            out = self.channel_mixer(out.transpose(1, 2)).transpose(1, 2)
        if self.use_post_conv:
            out = out + self.post_conv(out.transpose(1, 2)).transpose(1, 2)
        # Channel-wise output constraints:
        # 0/1: PV & wind in [0, 1]; others: non-negative
        if out.size(-1) >= 2:
            out_pv = torch.sigmoid(out[..., 0])
            out_wind = torch.sigmoid(out[..., 1])
            if out.size(-1) > 2:
                out_rest = F.relu(out[..., 2:])
                out = torch.cat([out_pv.unsqueeze(-1), out_wind.unsqueeze(-1), out_rest], dim=-1)
            else:
                out = torch.stack([out_pv, out_wind], dim=-1)
        else:
            out = F.relu(out)
        out = self.masker(out, mask)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        seq_len,
        cond_dim,
        model_dim=128,
        depth=4,
        heads=4,
        channels=4,
        dropout=0.1,
        patch_scales=3,
        high_weight=0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pre_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.in_proj = nn.Linear(channels, model_dim)
        self.time_proj = nn.Linear(2, model_dim)
        self.blocks = nn.ModuleList(
            [MHCBlock(model_dim, heads=heads, dropout=dropout, use_norm=False) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        conv_dim = max(16, model_dim // 4)
        self.channel_cond_proj = nn.Linear(cond_dim, conv_dim)
        self.cond_conv = nn.Sequential(
            nn.Conv1d(cond_dim, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.channel_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, conv_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                for _ in range(channels)
            ]
        )
        self.channel_heads = nn.ModuleList([nn.Linear(conv_dim, 1) for _ in range(channels)])
        self.diff_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.diff_cond_proj = nn.Linear(cond_dim, channels)
        self.diff_head = nn.Linear(channels, 1)
        self.high_weight = float(high_weight)
        self.patch_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.patch_scales = nn.ModuleList(
            [PatchScale(channels, cond_dim, hidden=8) for _ in range(patch_scales)]
        )
        self.register_buffer("time_feat", make_time_features(seq_len), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")

    def forward(self, x, cond):
        # x: (B, T, C), cond: (B, T, cond_dim) or (B, cond_dim)
        b, t, _ = x.shape
        if cond.dim() == 2:
            cond_mean = cond
            cond = cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        else:
            cond_mean = cond.mean(dim=1)
        x_raw = x
        x_mixed = self.pre_conv(x.transpose(1, 2)).transpose(1, 2)
        tfeat = self.time_proj(self.time_feat).unsqueeze(0).to(x.dtype)
        h = self.in_proj(x_mixed) + self.cond_proj(cond) + tfeat
        for blk in self.blocks:
            h = blk(h)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_pool = torch.cat([h_mean, h_max], dim=1)
        joint_scores = self.head(h_pool)

        cond_feat = self.channel_cond_proj(cond_mean)
        cond_seq_feat = self.cond_conv(cond.transpose(1, 2)).mean(dim=2)
        cond_feat = cond_feat + cond_seq_feat
        ch_scores = []
        for i, conv in enumerate(self.channel_convs):
            xi = x_raw[:, :, i].unsqueeze(1)  # (B, 1, T)
            feat = conv(xi).mean(dim=2)
            feat = feat + cond_feat
            ch_scores.append(self.channel_heads[i](feat))
        channel_scores = torch.cat(ch_scores, dim=1)
        x_s = x_raw.transpose(1, 2)
        c_s = cond.transpose(1, 2)
        patch_scores = []
        for i, patch in enumerate(self.patch_scales):
            if i > 0:
                x_s = self.patch_pool(x_s)
                c_s = self.patch_pool(c_s)
            patch_scores.append(patch(x_s.transpose(1, 2), c_s.transpose(1, 2)))
        patch_score = sum(patch_scores) / len(patch_scores) if patch_scores else 0.0
        channel_score = channel_scores.mean(dim=1, keepdim=True)
        if isinstance(patch_score, torch.Tensor):
            patch_score = patch_score.mean(dim=1, keepdim=True)
        out = joint_scores + channel_score + patch_score

        # High-frequency (diff) branch: x[t]-x[t-1]
        if x_raw.size(1) > 1 and self.high_weight != 0.0:
            x_diff = x_raw[:, 1:, :] - x_raw[:, :-1, :]
            diff_feat = self.diff_conv(x_diff.transpose(1, 2)).mean(dim=2)
            diff_feat = diff_feat + self.diff_cond_proj(cond_mean)
            diff_score = self.diff_head(diff_feat)
            out = out + self.high_weight * diff_score
        return out.squeeze(-1)


class PatchScale(nn.Module):
    def __init__(self, channels, cond_dim, hidden=8):
        super().__init__()
        width = channels * hidden
        self.dw_conv = nn.Conv1d(channels, width, kernel_size=3, padding=1, groups=channels)
        self.pw_conv = nn.Conv1d(width, channels, kernel_size=1, groups=channels)
        self.cond_conv = nn.Sequential(
            nn.Conv1d(cond_dim, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, cond):
        # x: (B, T, C), cond: (B, T, cond_dim)
        x_c = x.transpose(1, 2)
        feat = self.pw_conv(F.gelu(self.dw_conv(x_c)))
        score = feat.mean(dim=2)
        cond_feat = self.cond_conv(cond.transpose(1, 2)).mean(dim=2)
        score = score + cond_feat
        return score
