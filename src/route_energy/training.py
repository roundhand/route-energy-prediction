"""Joint-loss training with route-RMSE checkpoint selection."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from .models.baselines import DNNSD, DavisFCNet, DavisRET
from .models.proposed import RouteEnergyTransformer
from .metrics import metrics
from .utils import set_seed


def build(name, p, metadata, ef_metadata, sd, **model_options):
    (s, r) = (len(metadata.seg_feat_cols), len(metadata.route_feat_cols))
    if name == "DNN":
        return DNNSD(sd[0][0].shape[1], p["hidden1"], p["hidden2"], p["dropout"])
    if name == "FCNet":
        return DavisFCNet(s, r)
    if name == "RET":
        return DavisRET(s, r, p["n_blocks"], p["d_model"], dropout=p["dropout"])
    if name != "Proposed-EF":
        raise ValueError(f"Unsupported neural model: {name}")
    return RouteEnergyTransformer(
        ef_metadata.group_dims,
        ef_metadata.group_indices,
        r,
        **{k: p[k] for k in ["d_model", "n_heads", "n_layers", "d_ff", "dropout"]},
        **model_options,
    )


def forward(model, batch, name, device):
    if name == "DNN":
        return (None, model(batch[0].to(device)), batch[1].to(device), None, None)
    b = {k: v.to(device) for (k, v) in batch.items()}
    args = [b[k] for k in ["seg_seq", "route_ctx", "cum_dist", "mask"]]
    if name == "Proposed-EF":
        args.append(b["seg_lengths"])
    (seg, route) = model(*args)
    return (seg, route, b["label"], b["seg_labels"], b["mask"])


@torch.no_grad()
def predict(model, loader, name, device):
    model.eval()
    (ys, ps) = ([], [])
    for batch in loader:
        (_, pred, y, _, _) = forward(model, batch, name, device)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps))


def fit_nn(
    name,
    p,
    seed,
    data,
    epochs,
    patience,
    device,
    metadata,
    ef_metadata,
    sd,
    trial=None,
    *,
    model_options=None,
):
    set_seed(seed)
    model = build(name, p, metadata, ef_metadata, sd, **(model_options or {})).to(device)
    g = torch.Generator().manual_seed(seed)
    loaders = [
        DataLoader(
            d,
            batch_size=p["batch_size"],
            shuffle=i == 0,
            generator=g if i == 0 else None,
            num_workers=0,
        )
        for (i, d) in enumerate(data)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    (best, state, no_imp, best_epoch) = (float("inf"), None, 0, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders[0]:
            if name == "DNN" and batch[0].shape[0] < 2:
                continue
            (seg, pred, y, labels, mask) = forward(model, batch, name, device)
            loss = F.mse_loss(pred, y)
            if seg is not None:
                loss = loss + p["seg_loss_w"] * F.mse_loss(seg[mask], labels[mask])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        (yv, pv) = predict(model, loaders[1], name, device)
        score = metrics(yv, pv)["RMSE"]
        if score < best:
            (best, no_imp, best_epoch) = (score, 0, epoch)
            state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
        else:
            no_imp += 1
        if trial is not None:
            trial.report(best, epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if epoch % 10 == 0:
            print(
                f"  {name} seed={seed} epoch={epoch} val_RMSE={score:.6f} best={best:.6f}",
                flush=True,
            )
        if no_imp >= patience:
            break
    model.load_state_dict(state)
    return (model, loaders, best, best_epoch)


def fit_tab(name, p, seed, agg):
    if name == "Distance-based Regression":
        return LinearRegression(**p).fit(agg[0][0], agg[0][2])
    if name == "XGBoost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            **p,
            random_state=seed,
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            objective="reg:squarederror",
        )
        model.fit(agg[0][1], agg[0][2], verbose=False)
    else:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(**p, random_state=seed, verbosity=-1, n_jobs=4, subsample_freq=1)
        model.fit(agg[0][1], agg[0][2])
    return model


def tab_predict(model, name, item):
    p = model.predict(item[0 if name == "Distance-based Regression" else 1])
    return p if name == "Distance-based Regression" else np.maximum(p, 0)
