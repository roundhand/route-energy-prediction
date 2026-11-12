# Input data specification

Inputs are **precomputed segment and route features**, not raw GPS trajectories. Feature extraction from raw trajectories and road networks is outside this repository.

## Directory layout

```text
data_root/
  day_2023-09-11/
    segment_features_output/
      segment_features_clean_batch_0.csv
      segment_features_clean_batch_1.csv
    features/
      poi_feature_0.csv
  day_2023-09-12/
    ...
```

Each configured day must contain at least one segment file matching `segment_features_clean_batch_*.csv` and one route file matching `poi_feature_*.csv`. UTF-8 CSV with a header row is recommended. File names are sorted before concatenation.

## Segment CSV columns

| Column | Meaning / representation |
|---|---|
| `target` | Vehicle/entity identifier; must match route CSV |
| `moveid` | Trip identifier within a day; must match route `moveid_int` |
| `entry_time` | Sortable timestamp defining segment order |
| `length` | Segment distance in **meters** |
| `energy_total` | Observed segment energy in **kWh**, the regression label |
| `traffic_speed_mean`, `traffic_speed_std` | Precomputed traffic speed statistics |
| `traffic_acceleration_mean`, `traffic_acceleration_std` | Precomputed acceleration statistics |
| `traffic_strong_dec_ratio`, `traffic_dec_ratio` | Deceleration proportions |
| `traffic_steady_ratio`, `traffic_acc_ratio`, `traffic_strong_acc_ratio` | Remaining driving-state proportions |
| `pseudo_acceleration` | Precomputed spatial speed variation feature |
| `slope_mean`, `slope_std` | Precomputed slope statistics |
| `entry_angle`, `exit_angle` | Original angle convention; code transforms these to `180 - angle` |
| `road_type` | Categorical road class |
| `entry_turn_type`, `exit_turn_type` | Categorical turning classes |

Keep feature definitions and units consistent across splits. The historical preprocessing pipeline is not included, so the source physical convention for speed, acceleration and slope must be maintained by the data provider. Numeric feature NaNs and energy-label NaNs follow the original fill-with-zero policy. Segment duplicates are removed on `(target, moveid, entry_time)`.

## Route CSV columns

| Column | Meaning / accepted values |
|---|---|
| `target` | Matching vehicle/entity identifier |
| `moveid_int` | Trip identifier; renamed to `moveid` by the loader |
| `stime` | Trip start timestamp; used for daily weather lookup |
| `POI_time_sensitivity` | `不敏感`, `敏感`, `高敏感`; records labeled `未知` are removed |
| `rush_hour` | Original categories: `平峰`, `早高峰`, `晚高峰` |

Optional `moveid_x` and `moveid_y` columns are discarded. Provide a unique route-context row per `(target, moveid_int)` to avoid duplicate joins. Trip identifiers are prefixed with `YYYYMMDD_` to distinguish days. Segment trips without matching route context are removed.

Weather is supplied in configuration, for example:

```json
{
  "start_date": "2023-09-11",
  "end_date": "2023-09-15",
  "weather_map": {
    "2023-09-11": "小雨",
    "2023-09-12": "小雨",
    "2023-09-13": "中雨",
    "2023-09-14": "小雨",
    "2023-09-15": "阴天"
  }
}
```

Weather mapping must cover all retained route dates. Road type, turning, rush-hour and weather are one-hot encoded; POI sensitivity is mapped to 0/1/2. To preserve the completed experiment, categorical vocabulary is derived before the split, whereas numerical scalers fit training rows only. See `feature_metadata.json` in generated outputs for exact feature order.

## Dataset tensors and labels

- `seg_seq`: `[40, segment_feature_count]`; sorted, truncated or padded.
- `route_ctx`: `[route_feature_count]`; POI, rush-hour and weather context.
- `cum_dist`: `[40]`; cumulative raw segment distance in meters.
- `mask`: `[40]`; true for retained segments, false for padding.
- `label`: sum of retained segment energy, in kWh.
- Baseline `seg_labels`: segment energy in kWh.
- EF `seg_labels`: segment energy divided by length, in kWh/m.
- EF `seg_lengths`: distance used in `route_energy = sum(predicted_EF * length)`; minimum 1 meter, matching the original implementation.

Baseline and EF path labels are explicitly checked for equality. Long routes represent their first 40 segments, not full-trip energy beyond that prefix.

## Public demo

`python -m route_energy.synthetic --output data/demo` creates all required columns using random, synthetic values and synthetic identifiers. It does not derive from real records and does not approximate the distribution of the private benchmark.
