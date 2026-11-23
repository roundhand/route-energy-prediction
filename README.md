# Route Energy Benchmark

[中文说明](README.zh-CN.md)

EV route energy prediction with **joint-loss training and unified route-level validation RMSE** for hyperparameter selection, neural early stopping and checkpoint selection. This repository packages a completed research experiment as standalone Python modules; no notebook is required at runtime.

Models: distance-only linear regression, XGBoost, LightGBM, static–dynamic DNN, FCNet, GRU, RET and a grouped EF transformer (Proposed-EF).

## Install

Python 3.9–3.12. Use a virtual environment, then run from the repository root:

```bash
python -m pip install -e ".[dev]"
```

For CUDA, install the appropriate PyTorch 2.5 build for your machine first. The completed reference run used Python 3.9 and PyTorch 2.5.1+cu121. Historical dependency versions are listed in `requirements-reference.txt`; this file records that environment rather than a cross-platform lockfile.

## Quick smoke test without private data

```bash
python -m route_energy.synthetic --output data/demo
python -m route_energy --data-root data/demo --output outputs/demo --models "FCNet" "Proposed-EF" --trials 1 --search-epochs 1 --final-epochs 1 --seeds 1 --device cpu
python -m pytest -q
```

Synthetic data verifies execution only; its metrics are not scientific benchmark results.

## Full experiment

Prepare data following [the CSV specification](docs/data-format.md), then:

```bash
python -m route_energy --config configs/experiment.json --data-root /path/to/preprocessed/data
```

Windows paths also work: `--data-root "D:/datasets/ev_energy"`. Paths in JSON are resolved relative to the current working directory. Change the date range and `weather_map` together for another dataset. The original raw dataset is **not distributed**. Reference results therefore require equivalent privately held data to reproduce.

Defaults are **30 search trials per stochastic model**, up to 50 epochs per trial (patience 10), followed by **20 final seeds**, up to 100 epochs (patience 15). This is a long-running workload; smoke-test budgets are shown above. Linear regression exhaustively searches two intercept options and is fitted once.

Re-run the identical command to resume completed trials and final seeds. Use a new output directory when changing configuration or source. Only one writer may use an output directory. If a process was force-killed, verify it has stopped before removing its stale `.run.lock`. Data files must remain unchanged while resuming. Sampler RNG state is not serialized; the remaining search trajectory can differ after a restart.

## Ablation and performance distributions

```bash
python -m pip install -e ".[ablation]"
python -m route_energy.ablation --data-root "D:/datasets/ev_energy" --device cuda
```

The full model and three ablations use the selected Proposed-EF hyperparameters and the main benchmark's split, seed list, and validation-RMSE checkpoint rule. `w/o Traffic Stability` removes the traffic-stability feature group. `w/o Distance-Aware` removes the cumulative-distance encoding and attention bias while retaining learned segment order embeddings and FiLM. `w/o Both` combines these two removals. All four configurations use the grouped encoder.

To reuse the full model's existing seed results, pass `--reference-run` with the matching benchmark output directory, including its split IDs and per-seed predictions. Runs resume from `outputs/ablation-two-factor/results.json`. The completed run exports a three-panel figure (PNG and SVG), its caption, and per-seed and summary CSVs with MAE and RMSE in Wh and MAPE in percent.

## Method

| Model | Input | Training target |
|---|---|---|
| Distance regression | Total route distance | Route energy |
| XGBoost / LightGBM | Segment mean/sum/std + distance/count + route context | Route energy |
| DNN | Segment mean/std/min/max/sum/last + distance/count + route context | Route energy |
| FCNet | Independent segment features + route context | Segment and route energy |
| GRU / RET | Ordered segment sequence + route context | Segment and route energy |
| Proposed-EF | Grouped segment features + route context + distance information | Segment EF and route energy |

Route context includes POI sensitivity, rush-hour category and weather. All splits are fixed by route ID, with random state 42. Neural models with segment outputs retain `route_MSE + seg_loss_w * segment_MSE`; Proposed-EF uses segment EF MSE. Only validation **route RMSE in kWh** chooses hyperparameters, pruning, early stopping and checkpoints. DNN uses route MSE; trees retain their regression objectives. Tree predictions retain the original nonnegative clipping policy.

At most the first **40 segments** of each route are used; the evaluated target is the sum of energy for those retained segments. Scalers fit training data only. RET uses the source experiment's joint MSE convention rather than an exact reproduction of a literature MAE objective. See [reproducibility notes](docs/reproducibility.md) for limitations and search spaces.

## Outputs and recorded results

New runs write to `outputs/route_rmse/` by default:

- `report.md`, `summary.csv`: completed-seed aggregate metrics and best parameters.
- `results.json`: per-seed metrics, trainable parameter counts and new-run timing.
- `*_best_params.json`, `*_trials.csv`, `studies.sqlite3`: search parameters and trial history.
- `*_seed_*.pt`, `*_predictions.npz`: neural state dictionaries and test predictions (`y_true`, `y_pred`).
- `*_ids.csv`, `feature_metadata.json`, `scalers.pkl`: local split/preprocessing artifacts. Load pickle files only from a source you trust.

New-run `train_seconds` includes model construction, fitting and epoch validation. `run_seconds` additionally includes final validation/test prediction, but excludes artifact serialization. Timers synchronize CUDA; they measure elapsed wall time, not pure GPU compute. Aggregate training time excludes shared preprocessing. Historical reference runs have no per-seed training timers; missing values are not reconstructed.

The completed experiment's **metrics and parameters only** are in [results/reference/report.md](results/reference/report.md). No raw records, route IDs, per-route predictions, trained weights, logs or Optuna database from that experiment are included. `results/reference/summary.csv` includes neural parameter counts; tree parameter counts remain unrecorded.

Regenerate a report:

```bash
python -m route_energy.report results/reference
```

## Repository layout

```text
configs/                     JSON experiment configuration
src/route_energy/
  data.py                    CSV preparation, datasets, fixed splits, aggregation
  models/                    Neural baselines and proposed EF transformer
  training.py                Optimization and best-checkpoint restoration
  search.py                  Independent model search spaces
  experiment.py              CLI, Optuna, multi-seed evaluation, resume and timing
  metrics.py, report.py       Route metrics and summary exports
  synthetic.py               Public synthetic-data generator
tests/                       Integration and protocol checks
docs/                        Data schema and reproducibility details
results/reference/           Recorded metrics and best parameters
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Code is provided under the [MIT License](LICENSE). Dataset access and redistribution rights are separate from the code license. Model labels and provenance follow the source experiment; see [reproducibility notes](docs/reproducibility.md).
