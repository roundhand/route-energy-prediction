"""Baseline architectures preserved from the completed experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 40


class DNNSD(nn.Module):
    """
    DNN-SD (Static-Dynamic DNN) —— 文献 baseline，不是 Proposed 架构。
    普通前馈 DNN，输入 = [static route feature | dynamic segment-summary feature]：
      Input → Linear → ReLU → BatchNorm1d → Dropout → Linear → ReLU → Dropout → Linear(1)
    """

    def __init__(self, in_dim, hidden1=128, hidden2=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DavisFCNet(nn.Module):
    """
    Davis et al. (2024) Fully Connected NN baseline.
    每个路段独立：seg_feat ⊕ route_ctx → FC(32) → ReLU → FC(32) → ReLU → FC(1) → softplus → 求和
    """

    def __init__(self, seg_dim, route_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seg_dim + route_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seg_seq, route_ctx, cum_dist, mask):
        (B, L, _) = seg_seq.shape
        route_exp = route_ctx.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([seg_seq, route_exp], dim=-1)
        seg_energy = F.softplus(self.net(x).squeeze(-1))
        seg_energy = seg_energy * mask.float()
        route_energy = seg_energy.sum(dim=-1)
        return (seg_energy, route_energy)


class RETBlock(nn.Module):
    """GPT2-style pre-norm transformer block with causal self-attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask, pad_mask):
        h = self.ln1(x)
        (attn_out, _) = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=pad_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class DavisRET(nn.Module):
    """
    Davis et al. (2024) Route Energy Transformer.
    Decoder-only (causal) transformer — GPT2 风格：
      seg_feat ⊕ route_ctx → Linear → + learned pos_embed
      → N × (LayerNorm → Causal Self-Attn → Residual → LayerNorm → FFN → Residual)
      → LayerNorm → Linear(1) → softplus → sum
    """

    def __init__(self, seg_dim, route_dim, n_blocks, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        n_heads = max(1, d_model // 32)
        d_ff = d_model * 4
        self.input_proj = nn.Linear(seg_dim + route_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [RETBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, seg_seq, route_ctx, cum_dist, mask):
        (B, L, _) = seg_seq.shape
        route_exp = route_ctx.unsqueeze(1).expand(-1, L, -1)
        x = self.input_proj(torch.cat([seg_seq, route_exp], dim=-1))
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.drop(x + self.pos_embed(pos))
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        pad_mask = ~mask
        for block in self.blocks:
            x = block(x, causal_mask, pad_mask)
        x = self.ln_f(x)
        seg_energy = F.softplus(self.head(x).squeeze(-1))
        seg_energy = seg_energy * mask.float()
        route_energy = seg_energy.sum(dim=-1)
        return (seg_energy, route_energy)
