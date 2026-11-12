"""Grouped EF transformer; predicts EF and sums EF times segment length."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 40


class GroupMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RoadConditionEncoder(nn.Module):
    def __init__(self, slope_dim, turn_dim, type_dim, d_model, dropout=0.1):
        super().__init__()
        self.slope_mlp = GroupMLP(slope_dim, d_model, dropout)
        self.turn_mlp = GroupMLP(turn_dim, d_model, dropout)
        self.type_mlp = GroupMLP(type_dim, d_model, dropout)
        self.fusion = nn.Linear(3 * d_model, d_model)

    def forward(self, slope, turn, rtype):
        return self.fusion(
            torch.cat([self.slope_mlp(slope), self.turn_mlp(turn), self.type_mlp(rtype)], dim=-1)
        )


class SegmentGroupEncoder(nn.Module):
    def __init__(self, group_dims, group_indices, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.idx = group_indices
        self.traffic_mlp = GroupMLP(group_dims["traffic_state"], d_model, dropout)
        self.stability_mlp = GroupMLP(group_dims["traffic_stability"], d_model, dropout)
        self.road_enc = RoadConditionEncoder(
            group_dims["slope"], group_dims["turning"], group_dims["road_type"], d_model, dropout
        )
        self.group_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        (B, L, _) = x.shape
        idx = self.idx
        group_embs = [
            self.traffic_mlp(x[..., idx["traffic_state"]]),
            self.stability_mlp(x[..., idx["traffic_stability"]]),
            self.road_enc(x[..., idx["slope"]], x[..., idx["turning"]], x[..., idx["road_type"]]),
        ]
        groups = torch.stack(group_embs, dim=2).view(B * L, 3, -1)
        query = self.agg_query.expand(B * L, -1, -1)
        (out, _) = self.group_attn(query, groups, groups)
        return self.norm(out.squeeze(1)).view(B, L, -1)


class DistancePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, max_dist_m=60000):
        super().__init__()
        self.max_dist_m = max_dist_m
        self.order_embed = nn.Embedding(max_len, d_model)
        pe = torch.zeros(1000, d_model)
        pos = torch.arange(1000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("dist_pe", pe)

    def forward(self, x, cum_dist):
        (B, L, _) = x.shape
        order_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        dist_idx = (cum_dist / self.max_dist_m * 999).clamp(0, 999).long()
        return x + self.order_embed(order_idx) + self.dist_pe[dist_idx]


class DistanceBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.slopes = nn.Parameter(torch.rand(n_heads) * 0.1 + 0.05)

    def forward(self, cum_dist):
        d = (cum_dist.unsqueeze(2) - cum_dist.unsqueeze(1)).abs() / 1000.0
        return -self.slopes.abs().view(-1, 1, 1) * d.unsqueeze(1)


class FiLM(nn.Module):
    def __init__(self, route_dim, d_model):
        super().__init__()
        self.gamma = nn.Linear(route_dim, d_model)
        self.beta = nn.Linear(route_dim, d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, route_emb):
        return self.gamma(route_emb).unsqueeze(1) * x + self.beta(route_emb).unsqueeze(1)


class SegmentTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, route_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.film = FiLM(route_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, route_emb, dist_bias, pad_mask):
        (B, L, _) = x.shape
        h = self.norm1(x)
        Q = self.q(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale + dist_bias
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        x = x + self.drop(self.out_proj(out))
        x = x + self.drop(self.ff(self.norm2(x)))
        return self.film(x, route_emb)


class RouteEnergyTransformer(nn.Module):
    def __init__(
        self,
        group_dims,
        group_indices,
        route_dim,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        max_len=MAX_LEN,
    ):
        super().__init__()
        self.seg_encoder = SegmentGroupEncoder(group_dims, group_indices, d_model, n_heads, dropout)
        self.route_proj = nn.Linear(route_dim, d_model)
        self.pos_enc = DistancePositionalEncoding(d_model, max_len)
        self.dist_bias = DistanceBias(n_heads)
        self.layers = nn.ModuleList(
            [
                SegmentTransformerLayer(d_model, n_heads, d_ff, dropout, d_model)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, 1)
        )

    def forward(self, seg_seq, route_ctx, cum_dist, mask, seg_lengths):
        route_emb = self.route_proj(route_ctx)
        x = self.seg_encoder(seg_seq)
        x = self.pos_enc(x, cum_dist)
        dist_bias = self.dist_bias(cum_dist)
        pad_mask = ~mask
        for layer in self.layers:
            x = layer(x, route_emb, dist_bias, pad_mask)
        x = self.norm(x)
        seg_ef = F.softplus(self.seg_head(x).squeeze(-1)) * mask.float()
        route_energy = (seg_ef * seg_lengths).sum(dim=-1)
        return (seg_ef, route_energy)
