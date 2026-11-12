"""Command-line entry point for search, multi-seed refitting and reporting."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .data import prepare
from .metrics import metrics
from .report import export_results
from .search import suggest
from .training import fit_nn, fit_tab, tab_predict, predict
from .utils import atomic_json

MODELS = [
    "Distance-based Regression",
    "XGBoost",
    "LightGBM",
    "DNN",
    "FCNet",
    "GRU",
    "RET",
    "Proposed-EF",
]
SEEDS = [1, 7, 13, 21, 42, 55, 77, 88, 99, 137, 168, 233, 256, 314, 377, 404, 512, 610, 987, 999]
DEFAULTS = dict(
    data_root="data/raw",
    output="outputs/route_rmse",
    start_date="2023-09-11",
    end_date="2023-09-15",
    trials=30,
    search_epochs=50,
    final_epochs=100,
    seeds=20,
    models=MODELS,
    device="auto",
    weather_map={
        "2023-09-11": "小雨",
        "2023-09-12": "小雨",
        "2023-09-13": "中雨",
        "2023-09-14": "小雨",
        "2023-09-15": "阴天",
    },
)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration; CLI options override it")
    for name in ["data-root", "output", "start-date", "end-date"]:
        parser.add_argument("--" + name)
    for name in ["trials", "search-epochs", "final-epochs", "seeds"]:
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = vars(parser.parse_args(argv))
    config = dict(DEFAULTS)
    path = args.pop("config")
    if path:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(supplied)
    config.update({k: v for k, v in args.items() if v is not None})
    for key in ["trials", "search_epochs", "final_epochs", "seeds"]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if config["seeds"] > len(SEEDS):
        parser.error("seeds must be at most 20")
    if not config["models"] or any(m not in MODELS for m in config["models"]):
        parser.error("models must be a nonempty subset of supported model names")
    if config["device"] not in ["auto", "cpu", "cuda"]:
        parser.error("device must be auto, cpu or cuda")
    if pd.Timestamp(config["start_date"]) > pd.Timestamp(config["end_date"]):
        parser.error("start_date must not follow end_date")
    config.update(
        selection_metric="route_RMSE_kWh",
        search_seed=42,
        seeds_used=SEEDS[: config["seeds"]],
        split_random_state=42,
        max_len=40,
        search_patience=10,
        final_patience=15,
    )
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output"] = str(Path(config["output"]).resolve())
    return config


@contextmanager
def output_lock(out):
    """Prevent simultaneous writers. A hard-killed run may leave a stale lock."""
    lock = out / ".run.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock} exists. Check for a running writer before removing a stale lock."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        _run_locked(config, out)


def _run_locked(config, out):
    torch.set_num_threads(4)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    device = torch.device(device_name)
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "models"} != {
            k: v for k, v in config.items() if k != "models"
        }:
            raise ValueError("Experiment configuration changed; choose a new output directory.")
    atomic_json(config_path, config)
    print(f"Device: {device}; torch={torch.__version__}", flush=True)
    metadata, ef_metadata, bd, ed, agg, sd, source_hash = prepare(
        out, config["data_root"], config["start_date"], config["end_date"], config["weather_map"]
    )
    manifest = dict(
        source_hash=source_hash,
        counts=[len(d) for d in bd],
        dataset_labels="sum of first at most 40 segment energy labels",
        torch_version=torch.__version__,
        device=str(device),
    )
    manifest_path = out / "data_manifest.json"
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old_manifest["source_hash"] != source_hash
            or old_manifest["counts"] != manifest["counts"]
        ):
            raise ValueError("Source or data counts changed; use a new output directory.")
    atomic_json(manifest_path, manifest)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in config["models"]:
        slug = name.replace(" ", "_")
        is_tab = name in MODELS[:3]
        data = (
            [TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) for x, y in sd]
            if name == "DNN"
            else ed
            if name == "Proposed-EF"
            else bd
        )
        sampler = (
            optuna.samplers.GridSampler({"fit_intercept": [True, False]}, seed=42)
            if name == MODELS[0]
            else optuna.samplers.TPESampler(seed=42)
        )
        study = optuna.create_study(
            direction="minimize",
            study_name=slug,
            storage="sqlite:///" + (out / "studies.sqlite3").as_posix(),
            load_if_exists=True,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        # With the exclusive writer lock, RUNNING entries can only be interrupted old trials.
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        def objective(trial):
            p = suggest(trial, name)
            trial.set_user_attr("effective_params", p)
            synchronize(device)
            start = time.perf_counter()
            try:
                if is_tab:
                    model = fit_tab(name, p, 42, agg)
                    return metrics(agg[1][2], tab_predict(model, name, agg[1]))["RMSE"]
                _, _, score, _ = fit_nn(
                    name,
                    p,
                    42,
                    data,
                    config["search_epochs"],
                    10,
                    device,
                    metadata,
                    ef_metadata,
                    sd,
                    trial,
                )
                return score
            finally:
                synchronize(device)
                duration = time.perf_counter() - start
                trial.set_user_attr("elapsed_seconds", duration)
                print(f"{name} trial={trial.number} seconds={duration:.3f}", flush=True)

        budget = 2 if name == MODELS[0] else config["trials"]
        finished = sum(
            t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            for t in study.trials
        )
        remaining = max(0, budget - finished)
        print(f"\n{name}: {remaining} search trials remaining", flush=True)
        try:
            if remaining:
                study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        finally:
            study.trials_dataframe().to_csv(out / f"{slug}_trials.csv", index=False)
        p = study.best_trial.user_attrs["effective_params"]
        atomic_json(out / f"{slug}_best_params.json", p)
        rec = results.setdefault(
            name, {"best_params": p, "search_val_RMSE": study.best_value, "runs": []}
        )
        rec["search_elapsed_seconds"] = sum(
            t.user_attrs.get("elapsed_seconds", 0) for t in study.trials
        )
        if name == MODELS[0] and rec["runs"]:
            continue
        for seed in config["seeds_used"]:
            if any(r["seed"] == seed for r in rec["runs"]):
                continue
            synchronize(device)
            start = time.perf_counter()
            if is_tab:
                model = fit_tab(name, p, seed, agg)
                synchronize(device)
                train_seconds = time.perf_counter() - start
                pv, pt = tab_predict(model, name, agg[1]), tab_predict(model, name, agg[2])
                yv, yt = agg[1][2], agg[2][2]
                epoch, nparams = (
                    None,
                    (2 if p["fit_intercept"] else 1) if name == MODELS[0] else None,
                )
            else:
                model, loaders, _, epoch = fit_nn(
                    name,
                    p,
                    seed,
                    data,
                    config["final_epochs"],
                    15,
                    device,
                    metadata,
                    ef_metadata,
                    sd,
                )
                synchronize(device)
                train_seconds = time.perf_counter() - start
                yv, pv = predict(model, loaders[1], name, device)
                yt, pt = predict(model, loaders[2], name, device)
                nparams = sum(v.numel() for v in model.parameters() if v.requires_grad)
            synchronize(device)
            run_seconds = time.perf_counter() - start
            rec["runs"].append(
                dict(
                    seed=seed,
                    best_epoch=epoch,
                    n_params=nparams,
                    train_seconds=train_seconds,
                    run_seconds=run_seconds,
                    validation=metrics(yv, pv),
                    test=metrics(yt, pt),
                )
            )
            if not is_tab:
                torch.save(model.cpu().state_dict(), out / f"{slug}_seed_{seed}.pt")
            np.savez_compressed(out / f"{slug}_seed_{seed}_predictions.npz", y_true=yt, y_pred=pt)
            for metric in ["MAE", "RMSE", "MAPE%", "R2"]:
                values = [r["test"][metric] for r in rec["runs"]]
                rec[metric + "_mean"] = float(np.mean(values))
                rec[metric + "_std"] = float(np.std(values))
            atomic_json(results_path, results)
            export_results(results, out)
            print(
                f"FINAL {name} seed={seed}: {rec['runs'][-1]['test']} train_seconds={train_seconds:.3f} params={nparams}",
                flush=True,
            )
            if name == MODELS[0]:
                break
    export_results(results, out)
    print(f'Finished. Results: {out / "report.md"}', flush=True)


def main(argv=None):
    run(parse_config(argv))


if __name__ == "__main__":
    main()
