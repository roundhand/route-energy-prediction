# Experiment provenance and reproducibility

## Source and preserved behavior

This package was extracted from the completed local unified-RMSE experiment. Baseline model/data definitions originally lived in `8.19 baseline_FCNet.ipynb`; the EF transformer lived in `8.13 EF 调参.ipynb`. The orchestrator was `tune_baselines.py`. The distribution contains standalone definitions, not these source notebooks or a runtime notebook executor.

Model names such as DavisFCNet, DavisGRU and DavisRET preserve the source experiment's labels. The implementations are research adaptations. In particular, RET uses the experiment's joint MSE training loss, not an exact reproduction of a paper's MAE training objective, and tuning changes its block count and width. No new bibliographic metadata or paper citation is invented here.

Reference data covered September 11–15, 2023. Retained split sizes were 32,602 training, 4,071 validation and 4,075 test routes. Splitting is performed on `(target, moveid)` with `train_test_split(test_size=0.1, random_state=42)`, followed by `test_size=0.111` for validation. This is a random route split; it is not a held-out-vehicle or chronological split.

All models use identical split IDs and retained energy labels. Neural segment-output models keep joint training loss; only route validation RMSE determines hyperparameters and checkpoints. Test metrics are not used for selection. Route context scalers follow the source convention of fitting context values repeated across training segment rows. Aggregate DNN features receive an additional scaler fitted on training routes.

## Search spaces

All stochastic searches use seeded TPE (seed 42), 30 trials, and a median pruner with 5 startup trials and 10 warmup steps. COMPLETE and PRUNED trials count toward the budget. NN training uses AdamW, cosine learning-rate scheduling, gradient clipping at 1.0, search patience 10 and final patience 15.

| Models | Hyperparameters |
|---|---|
| Distance regression | `fit_intercept`: true or false, exhaustive |
| XGBoost / LightGBM | estimators: 200–1000 step 200; learning rate: 0.01–0.2 log; depth: 3–10; row/column sampling: 0.6–1.0; L1/L2: 1e-5–10 log |
| XGBoost additional | minimum child weight: 1–20 log |
| LightGBM additional | leaves: 8–min(128, 2^depth); minimum child samples: 10–80; subsample frequency 1 |
| All NN models | learning rate: 1e-4–3e-3 log; weight decay: 1e-6–1e-2 log |
| Baseline NNs | batch size: 32, 64, 128 |
| Segment-output NNs | segment loss weight: 0.05–0.5 |
| DNN additional | first hidden: 64/128/256; second hidden: 32/64/128; dropout: 0.05–0.3 |
| RET additional | blocks: 1–3; model width: 32/64/128; dropout: 0.05–0.3 |
| Proposed-EF additional | width: 64/128/256; heads: 2/4/8; layers: 2–4; FF width: 128/256/512; dropout: 0.05–0.3; fixed batch size 64 |

FCNet and GRU retain source architecture dimensions and independently tune optimization/loss parameters. Trees use the full suggested estimator count; no internal early stopping is used. Tree predictions are clipped at zero. Baseline segment inputs include length; EF excludes length from segment feature vectors and uses it in distance encoding/bias and energy reconstruction.

Final seed list: `1, 7, 13, 21, 42, 55, 77, 88, 99, 137, 168, 233, 256, 314, 377, 404, 512, 610, 987, 999`. Population standard deviation uses `ddof=0`. MAPE is `mean(abs((prediction - target) / (target + 1e-8))) * 100`; near-zero targets can produce large values.

## Reference artifacts

`results/reference/results.json` contains recorded validation/test metrics per seed and trainable parameter counts, without individual route data. `best_params.json` contains selected settings. `summary.csv` and `report.md` are derived summaries. These are the results of the original completed run, not a fresh retraining by this package.

Historical per-seed training times were not recorded. They remain absent. New runs include synchronized elapsed timers for trials, fitting and evaluation, and include parameter counts in the summary. They also save preprocessing metadata/scalers, use an exclusive output lock and export an English report for arbitrary output directories. These are packaging/observability changes, not claims of identical timing or byte-for-byte training trajectories.

## Reproduction boundaries

- The original data and feature-generation pipeline are not public in this package. Exact metrics require equivalent data and preprocessing.
- Historical package versions are recorded in `requirements-reference.txt`; current installation uses compatible ranges. Hardware, package versions and random kernels can change results.
- The code sets deterministic cuDNN flags as in the source experiment, but does not promise bitwise reproducibility on all devices.
- Completed trials and seeds can resume. TPE sampler RNG state is not serialized, so continuation after a restart can differ from an uninterrupted run.
- The output source hash and row counts detect common changes; they are not a cryptographic fingerprint of every private input CSV. Do not change data in a reused output directory.
- CPU synthetic tests exercise all model families and the selection protocol; they do not replace full-scale scientific replication.
