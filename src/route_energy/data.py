"""Raw CSV loading, original feature processing, datasets and fixed splits."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

TARGET_COL = "energy_total"
MAX_LEN = 40


@dataclass
class FeatureMetadata:
    seg_feat_cols: list
    route_feat_cols: list
    group_dims: dict
    group_indices: dict


def load_raw(data_root, start_date, end_date):
    seg_parts, route_parts = [], []
    for dt in pd.date_range(start_date, end_date):
        day = Path(data_root) / dt.strftime("day_%Y-%m-%d")
        prefix = dt.strftime("%Y%m%d") + "_"
        segments = sorted(
            (day / "segment_features_output").glob("segment_features_clean_batch_*.csv")
        )
        routes = sorted((day / "features").glob("poi_feature_*.csv"))
        if not segments or not routes:
            raise FileNotFoundError(f"Missing segment or route CSV files under {day}")
        frame = pd.concat([pd.read_csv(f, low_memory=False) for f in segments], ignore_index=True)
        frame["moveid"] = prefix + frame["moveid"].astype(str)
        seg_parts.append(frame.drop_duplicates(subset=["target", "moveid", "entry_time"]))
        frame = pd.concat([pd.read_csv(f, low_memory=False) for f in routes], ignore_index=True)
        frame = frame.drop(columns=["moveid_x", "moveid_y"], errors="ignore")
        frame = frame.rename(columns={"moveid_int": "moveid"})
        frame["moveid"] = prefix + frame["moveid"].astype(str)
        route_parts.append(frame)
    if not seg_parts:
        raise ValueError("Date range is empty")
    return pd.concat(seg_parts, ignore_index=True), pd.concat(route_parts, ignore_index=True)


def preprocess(seg_df, route_df, weather_map):
    route_df["date"] = pd.to_datetime(route_df["stime"]).dt.normalize()
    route_df["weather"] = route_df["date"].map({pd.Timestamp(k): v for k, v in weather_map.items()})
    # 0 = 直行, 180 = 掉头, 越大表示转向扰动越强
    seg_df["entry_angle"] = 180 - seg_df["entry_angle"]
    seg_df["exit_angle"] = 180 - seg_df["exit_angle"]
    road_dummies = pd.get_dummies(seg_df["road_type"], prefix="road").astype("float32")
    seg_df = pd.concat([seg_df.reset_index(drop=True), road_dummies.reset_index(drop=True)], axis=1)
    ROAD_TYPE_COLS = list(road_dummies.columns)
    entry_turn_dummies = pd.get_dummies(seg_df["entry_turn_type"], prefix="entry_turn").astype(
        "float32"
    )
    exit_turn_dummies = pd.get_dummies(seg_df["exit_turn_type"], prefix="exit_turn").astype(
        "float32"
    )
    seg_df = pd.concat(
        [
            seg_df.reset_index(drop=True),
            entry_turn_dummies.reset_index(drop=True),
            exit_turn_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    ENTRY_TURN_COLS = list(entry_turn_dummies.columns)
    EXIT_TURN_COLS = list(exit_turn_dummies.columns)
    route_df = route_df[route_df["POI_time_sensitivity"] != "未知"].reset_index(drop=True)
    valid_ids = route_df[["target", "moveid"]].drop_duplicates()
    seg_df = seg_df.merge(valid_ids, on=["target", "moveid"], how="inner")
    poi_order = {"不敏感": 0, "敏感": 1, "高敏感": 2}
    route_df["POI_time_sensitivity"] = (
        route_df["POI_time_sensitivity"].map(poi_order).astype("float32")
    )
    rush_dummies = pd.get_dummies(route_df["rush_hour"], prefix="rush").astype("float32")
    route_df = pd.concat(
        [route_df.reset_index(drop=True), rush_dummies.reset_index(drop=True)], axis=1
    )
    RUSH_COLS = list(rush_dummies.columns)
    weather_dummies = pd.get_dummies(route_df["weather"], prefix="weather").astype("float32")
    route_df = pd.concat(
        [route_df.reset_index(drop=True), weather_dummies.reset_index(drop=True)], axis=1
    )
    WEATHER_COLS = list(weather_dummies.columns)
    if route_df["weather"].isna().any():
        raise ValueError("Weather mapping must cover all route dates")
    return (
        seg_df,
        route_df,
        ROAD_TYPE_COLS,
        ENTRY_TURN_COLS,
        EXIT_TURN_COLS,
        RUSH_COLS,
        WEATHER_COLS,
    )


def baseline_features(ROAD_TYPE_COLS, ENTRY_TURN_COLS, EXIT_TURN_COLS, RUSH_COLS, WEATHER_COLS):
    SEG_GROUP_COLS = {
        "basic": ["length"],
        "traffic_state": ["traffic_speed_mean", "traffic_speed_std"],
        "speed_oscillation": [
            "traffic_acceleration_mean",
            "traffic_acceleration_std",
            "traffic_strong_dec_ratio",
            "traffic_dec_ratio",
            "traffic_steady_ratio",
            "traffic_acc_ratio",
            "traffic_strong_acc_ratio",
        ],
        "spatial_variation": ["pseudo_acceleration"],
        "road_geometry": ["slope_mean", "slope_std", "entry_angle", "exit_angle"],
        "turning_behavior": ENTRY_TURN_COLS + EXIT_TURN_COLS,
        "road_type": ROAD_TYPE_COLS,
    }
    SEG_FEAT_COLS = [col for cols in SEG_GROUP_COLS.values() for col in cols]
    _col_to_idx = {c: i for (i, c) in enumerate(SEG_FEAT_COLS)}
    GROUP_INDICES = {g: [_col_to_idx[c] for c in cols] for (g, cols) in SEG_GROUP_COLS.items()}
    GROUP_DIMS = {g: len(cols) for (g, cols) in SEG_GROUP_COLS.items()}
    ROUTE_FEAT_COLS = ["POI_time_sensitivity", *RUSH_COLS, *WEATHER_COLS]
    TARGET_COL = "energy_total"
    DIST_COL = "length"
    MAX_LEN = 40
    return FeatureMetadata(SEG_FEAT_COLS, ROUTE_FEAT_COLS, GROUP_DIMS, GROUP_INDICES)


def ef_features(ROAD_TYPE_COLS, ENTRY_TURN_COLS, EXIT_TURN_COLS, RUSH_COLS, WEATHER_COLS):
    SEG_GROUP_COLS = {
        "traffic_state": ["traffic_speed_mean", "traffic_speed_std"],
        "traffic_stability": [
            "traffic_acceleration_mean",
            "traffic_acceleration_std",
            "traffic_strong_dec_ratio",
            "traffic_dec_ratio",
            "traffic_steady_ratio",
            "traffic_acc_ratio",
            "traffic_strong_acc_ratio",
            "pseudo_acceleration",
        ],
        "slope": ["slope_mean", "slope_std"],
        "turning": ["entry_angle", "exit_angle"] + ENTRY_TURN_COLS + EXIT_TURN_COLS,
        "road_type": ROAD_TYPE_COLS,
    }
    SEG_FEAT_COLS = [col for cols in SEG_GROUP_COLS.values() for col in cols]
    _col_to_idx = {c: i for (i, c) in enumerate(SEG_FEAT_COLS)}
    GROUP_INDICES = {g: [_col_to_idx[c] for c in cols] for (g, cols) in SEG_GROUP_COLS.items()}
    GROUP_DIMS = {g: len(cols) for (g, cols) in SEG_GROUP_COLS.items()}
    ROUTE_FEAT_COLS = ["POI_time_sensitivity", *RUSH_COLS, *WEATHER_COLS]
    TARGET_COL = "energy_total"
    MAX_LEN = 40
    return FeatureMetadata(SEG_FEAT_COLS, ROUTE_FEAT_COLS, GROUP_DIMS, GROUP_INDICES)


class BaselineTripDataset(Dataset):
    """
    每条样本 = 一次行程。
    seg_seq    : [MAX_LEN, seg_dim]  路段特征序列（不足补零）
    route_ctx  : [route_dim]         路径级特征（POI / 时段 / 天气）
    cum_dist   : [MAX_LEN]           各路段累积距离（米），用于距离位置编码
    seg_labels : [MAX_LEN]           各路段真实能耗（kWh），padding 位置为 0
    label      : scalar              行程总能耗（kWh）
    mask       : [MAX_LEN] bool      True = 有效路段
    """

    def __init__(
        self,
        seg_df,
        route_df,
        seg_feat_cols,
        route_feat_cols,
        target_col=TARGET_COL,
        max_len=MAX_LEN,
        seg_scaler=None,
        route_scaler=None,
        fit_scaler=True,
    ):
        self.max_len = max_len
        merged = seg_df.merge(
            route_df[["target", "moveid"] + route_feat_cols], on=["target", "moveid"], how="inner"
        )
        merged["_length_raw"] = merged["length"].fillna(0)
        seg_vals = merged[seg_feat_cols].fillna(0).values.astype("float32")
        route_vals = merged[route_feat_cols].fillna(0).values.astype("float32")
        if fit_scaler:
            self.seg_scaler = StandardScaler().fit(seg_vals)
            self.route_scaler = StandardScaler().fit(route_vals)
        else:
            self.seg_scaler = seg_scaler
            self.route_scaler = route_scaler
        merged[seg_feat_cols] = self.seg_scaler.transform(seg_vals)
        merged[route_feat_cols] = self.route_scaler.transform(route_vals)
        self.samples = []
        for ((target, moveid), grp) in merged.groupby(["target", "moveid"]):
            grp = grp.sort_values("entry_time").reset_index(drop=True)
            seg_seq = grp[seg_feat_cols].values.astype("float32")
            route_ctx = grp[route_feat_cols].iloc[0].values.astype("float32")
            lengths = grp["_length_raw"].values.astype("float32")
            cum_dist = np.cumsum(lengths)
            seg_labels = grp[target_col].fillna(0).values.astype("float32")
            self.samples.append((seg_seq, route_ctx, cum_dist, seg_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (seg_seq, route_ctx, cum_dist, seg_labels) = self.samples[idx]
        L = len(seg_seq)
        if L >= self.max_len:
            seg_seq = seg_seq[: self.max_len]
            cum_dist = cum_dist[: self.max_len]
            seg_labels = seg_labels[: self.max_len]
            label = seg_labels.sum()
            valid_L = self.max_len
        else:
            pad = self.max_len - L
            seg_seq = np.pad(seg_seq, ((0, pad), (0, 0)))
            cum_dist = np.pad(cum_dist, (0, pad))
            seg_labels = np.pad(seg_labels, (0, pad))
            label = seg_labels.sum()
            valid_L = L
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:valid_L] = True
        return {
            "seg_seq": torch.tensor(seg_seq, dtype=torch.float32),
            "route_ctx": torch.tensor(route_ctx, dtype=torch.float32),
            "cum_dist": torch.tensor(cum_dist, dtype=torch.float32),
            "seg_labels": torch.tensor(seg_labels, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


class EFTripDataset(Dataset):
    def __init__(
        self,
        seg_df,
        route_df,
        seg_feat_cols,
        route_feat_cols,
        target_col=TARGET_COL,
        max_len=MAX_LEN,
        seg_scaler=None,
        route_scaler=None,
        fit_scaler=True,
    ):
        self.max_len = max_len
        merged = seg_df.merge(
            route_df[["target", "moveid"] + route_feat_cols], on=["target", "moveid"], how="inner"
        )
        merged["_length_raw"] = merged["length"].fillna(0).clip(lower=1.0)
        seg_vals = merged[seg_feat_cols].fillna(0).values.astype("float32")
        route_vals = merged[route_feat_cols].fillna(0).values.astype("float32")
        if fit_scaler:
            self.seg_scaler = StandardScaler().fit(seg_vals)
            self.route_scaler = StandardScaler().fit(route_vals)
        else:
            self.seg_scaler = seg_scaler
            self.route_scaler = route_scaler
        merged[seg_feat_cols] = self.seg_scaler.transform(seg_vals)
        merged[route_feat_cols] = self.route_scaler.transform(route_vals)
        self.samples = []
        for ((target, moveid), grp) in merged.groupby(["target", "moveid"]):
            grp = grp.sort_values("entry_time").reset_index(drop=True)
            seg_seq = grp[seg_feat_cols].values.astype("float32")
            route_ctx = grp[route_feat_cols].iloc[0].values.astype("float32")
            lengths = grp["_length_raw"].values.astype("float32")
            cum_dist = np.cumsum(lengths)
            seg_energy = grp[target_col].fillna(0).values.astype("float32")
            seg_ef = seg_energy / lengths
            self.samples.append((seg_seq, route_ctx, cum_dist, lengths, seg_ef, seg_energy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (seg_seq, route_ctx, cum_dist, lengths, seg_ef, seg_energy) = self.samples[idx]
        L = len(seg_seq)
        if L >= self.max_len:
            (seg_seq, cum_dist, lengths) = (
                seg_seq[: self.max_len],
                cum_dist[: self.max_len],
                lengths[: self.max_len],
            )
            (seg_ef, seg_energy) = (seg_ef[: self.max_len], seg_energy[: self.max_len])
            valid_L = self.max_len
        else:
            pad = self.max_len - L
            seg_seq = np.pad(seg_seq, ((0, pad), (0, 0)))
            cum_dist = np.pad(cum_dist, (0, pad))
            lengths = np.pad(lengths, (0, pad))
            seg_ef = np.pad(seg_ef, (0, pad))
            seg_energy = np.pad(seg_energy, (0, pad))
            valid_L = L
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:valid_L] = True
        return {
            "seg_seq": torch.tensor(seg_seq, dtype=torch.float32),
            "route_ctx": torch.tensor(route_ctx, dtype=torch.float32),
            "cum_dist": torch.tensor(cum_dist, dtype=torch.float32),
            "seg_lengths": torch.tensor(lengths, dtype=torch.float32),
            "seg_labels": torch.tensor(seg_ef, dtype=torch.float32),
            "label": torch.tensor(seg_energy[mask].sum(), dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


class Materialized(Dataset):
    def __init__(self, dataset):
        samples = [dataset[i] for i in range(len(dataset))]
        self.data = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, i):
        return {k: v[i] for (k, v) in self.data.items()}


def extract_agg_features(dataset):
    """从 TripDataset 中提取聚合特征（用于行程级表格型 baseline）。"""
    (X_len, X_agg, y) = ([], [], [])
    for i in range(len(dataset)):
        s = dataset[i]
        seg = s["seg_seq"].numpy()
        rctx = s["route_ctx"].numpy()
        cd = s["cum_dist"].numpy()
        mask = s["mask"].numpy()
        label = s["label"].item()
        valid = seg[mask]
        total_dist = float(cd[mask][-1]) if mask.any() else 0.0
        n = max(len(valid), 1)
        seg_mean = valid.mean(axis=0) if len(valid) else np.zeros(seg.shape[1])
        seg_sum = valid.sum(axis=0) if len(valid) else np.zeros(seg.shape[1])
        seg_std = valid.std(axis=0) if len(valid) else np.zeros(seg.shape[1])
        X_len.append(total_dist)
        X_agg.append(np.concatenate([seg_mean, seg_sum, seg_std, [total_dist, n], rctx]))
        y.append(label)
    return (np.array(X_len).reshape(-1, 1), np.array(X_agg), np.array(y))


def extract_dnn_sd_features(dataset):
    """Static (route-level) + Dynamic (segment-level 统计聚合) 特征拼接。"""
    (X, y) = ([], [])
    for i in range(len(dataset)):
        s = dataset[i]
        seg = s["seg_seq"].numpy()
        rctx = s["route_ctx"].numpy()
        cd = s["cum_dist"].numpy()
        mask = s["mask"].numpy().astype(bool)
        label = s["label"].item()
        valid = seg[mask]
        total_dist = float(cd[mask][-1]) if mask.any() else 0.0
        n = max(len(valid), 1)
        if len(valid):
            seg_mean = valid.mean(axis=0)
            seg_std = valid.std(axis=0)
            seg_min = valid.min(axis=0)
            seg_max = valid.max(axis=0)
            seg_sum = valid.sum(axis=0)
            seg_last = valid[-1]
        else:
            zeros = np.zeros(seg.shape[1])
            seg_mean = seg_std = seg_min = seg_max = seg_sum = seg_last = zeros
        dynamic_summary = np.concatenate([seg_mean, seg_std, seg_min, seg_max, seg_sum, seg_last])
        static_feat = np.concatenate([rctx, [total_dist, n]])
        X.append(np.concatenate([static_feat, dynamic_summary]))
        y.append(label)
    return (np.array(X, dtype="float32"), np.array(y, dtype="float32"))


def prepare(out, data_root, start_date, end_date, weather_map):
    seg_df, route_df = load_raw(data_root, start_date, end_date)
    seg_df, route_df, *categories = preprocess(seg_df, route_df, weather_map)
    meta, ef_meta = baseline_features(*categories), ef_features(*categories)
    trip_ids = seg_df[["target", "moveid"]].drop_duplicates()
    train_ids, test_ids = train_test_split(trip_ids, test_size=0.1, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.111, random_state=42)
    ids = [train_ids, val_ids, test_ids]
    split_sets = [set(map(tuple, frame.values)) for frame in ids]
    if any(split_sets[i] & split_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Overlapping split IDs")
    baseline_ds, ef_ds = [], []
    for i, (name, split_ids) in enumerate(zip(["train", "val", "test"], ids)):
        split_ids.to_csv(out / f"{name}_ids.csv", index=False)
        frame = seg_df.merge(split_ids, on=["target", "moveid"])
        for cls, metadata, datasets in [
            (BaselineTripDataset, meta, baseline_ds),
            (EFTripDataset, ef_meta, ef_ds),
        ]:
            kwargs = {"fit_scaler": i == 0}
            if i:
                kwargs.update(
                    seg_scaler=datasets[0].seg_scaler, route_scaler=datasets[0].route_scaler
                )
            datasets.append(
                cls(frame, route_df, metadata.seg_feat_cols, metadata.route_feat_cols, **kwargs)
            )
    agg = [extract_agg_features(d) for d in baseline_ds]
    sd = [extract_dnn_sd_features(d) for d in baseline_ds]
    scaler = StandardScaler().fit(sd[0][0])
    sd = [(scaler.transform(x).astype("float32"), y) for x, y in sd]
    with (out / "scalers.pkl").open("wb") as f:
        pickle.dump(
            {
                "baseline_segment": baseline_ds[0].seg_scaler,
                "baseline_route": baseline_ds[0].route_scaler,
                "ef_segment": ef_ds[0].seg_scaler,
                "ef_route": ef_ds[0].route_scaler,
                "dnn_aggregate": scaler,
            },
            f,
        )
    (out / "feature_metadata.json").write_text(
        json.dumps(
            {"baseline": meta.__dict__, "ef": ef_meta.__dict__}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    bd, ed = [Materialized(d) for d in baseline_ds], [Materialized(d) for d in ef_ds]
    for a, z in zip(bd, ed):
        torch.testing.assert_close(a.data["label"], z.data["label"])
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return meta, ef_meta, bd, ed, agg, sd, digest.hexdigest()
