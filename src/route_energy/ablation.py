"""Refit the Proposed-EF ablations and export their test performance distributions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import prepare
from .experiment import DEFAULTS, SEEDS, output_lock
from .metrics import metrics
from .training import fit_nn, predict
from .utils import atomic_json

PROPOSED = "Proposed-EF"
VARIANTS = {
    PROPOSED: {},
    "w/o Traffic Stability": {"traffic_stability": False},
    "w/o Distance-Aware": {"distance_aware": False},
    "w/o Both": {"traffic_stability": False, "distance_aware": False},
}
METRICS = {"MAE": "MAE (Wh)", "RMSE": "RMSE (Wh)", "MAPE%": "MAPE (%)"}
SCALES = {"MAE": 1000, "RMSE": 1000, "MAPE%": 1}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_or_write(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"{path.name} changed. Choose a new output directory.")
    atomic_json(path, value)


def reference_runs(source, out, config, test_labels):
    """Reuse only a compatible full-model benchmark with per-seed predictions."""
    original_config = read_json(source / "config.json")
    for key in (
        "final_epochs",
        "final_patience",
        "selection_metric",
        "split_random_state",
        "max_len",
    ):
        if original_config[key] != config[key]:
            raise ValueError(f"Reference run used a different {key}.")
    for split in ("train", "val", "test"):
        filename = f"{split}_ids.csv"
        if (source / filename).read_text(encoding="utf-8-sig") != (out / filename).read_text(
            encoding="utf-8-sig"
        ):
            raise ValueError(f"Reference run has different or reordered {split} trip IDs.")
    proposed = read_json(source / "results.json")[PROPOSED]
    if proposed["best_params"] != config["params"]:
        raise ValueError("Reference Proposed-EF used different hyperparameters.")
    by_seed = {run["seed"]: run for run in proposed["runs"]}
    if len(by_seed) != len(proposed["runs"]) or not set(config["seeds_used"]) <= by_seed.keys():
        raise ValueError("Reference Proposed-EF runs have duplicate or missing seeds.")
    reused = []
    for seed in config["seeds_used"]:
        run = by_seed[seed]
        with np.load(source / f"{PROPOSED}_seed_{seed}_predictions.npz") as predictions:
            np.testing.assert_array_equal(predictions["y_true"], test_labels)
            recomputed = metrics(test_labels, predictions["y_pred"])
        np.testing.assert_allclose(
            list(recomputed.values()), [run["test"][k] for k in recomputed], rtol=1e-8, atol=1e-10
        )
        reused.append(dict(run, source="main benchmark"))
    return reused


def export_figure(results, seeds, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name in VARIANTS:
        runs = results[name]
        if sorted(run["seed"] for run in runs) != sorted(seeds):
            raise ValueError(f"Cannot plot {name}: expected one run for each requested seed.")
        for run in runs:
            rows.append(
                {
                    "Model": name,
                    "Seed": run["seed"],
                    **{label: run["test"][key] * SCALES[key] for key, label in METRICS.items()},
                    "Source": run["source"],
                }
            )
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "ablation_per_seed.csv", index=False, float_format="%.8f")
    summary = []
    for name, runs in scores.groupby("Model", sort=False):
        row = {"Model": name, "Seeds": len(runs), "Source": runs["Source"].iloc[0]}
        for label in METRICS.values():
            row[f"{label} mean"] = runs[label].mean()
            row[f"{label} std"] = runs[label].std(ddof=0)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "ablation_summary.csv", index=False, float_format="%.6f")

    ticks = [
        "Proposed-EF",
        "w/o Traffic\nStability",
        "w/o Distance-\nAware",
        "w/o Both",
    ]
    colors = ["#176a84", "#718e70", "#947659", "#737c87"]
    positions = np.arange(1, len(VARIANTS) + 1)
    with plt.rc_context(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), layout="constrained")
        for ax, panel, label in zip(axes, ("(a)", "(b)", "(c)"), METRICS.values()):
            values = [scores.loc[scores["Model"] == name, label].to_numpy() for name in VARIANTS]
            box = ax.boxplot(
                values,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#202830", "linewidth": 1.3},
            )
            for pos, patch, vals, color in zip(positions, box["boxes"], values, colors):
                patch.set(facecolor=color, alpha=0.23, edgecolor=color, linewidth=1.2)
                jitter = np.random.default_rng(pos).uniform(-0.23, 0.23, size=len(vals))
                ax.scatter(
                    pos + jitter, vals, s=13, color=color, alpha=0.75, linewidths=0, zorder=3
                )
            ax.set_xticks(positions, ticks, fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(panel, loc="left", fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
        fig.savefig(out / "ablation_distributions.png", dpi=300)
        fig.savefig(out / "ablation_distributions.svg")
        plt.close(fig)
    (out / "figure_caption.txt").write_text(
        "Figure. Performance distributions of the proposed model and ablated variants "
        f"across {len(seeds)} random seeds for (a) MAE, (b) RMSE, and (c) MAPE.\n",
        encoding="utf-8",
    )


def run(config):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch."
        )
    device = torch.device(device_name)
    torch.set_num_threads(4)
    seeds = config["seeds_used"]
    params = config["params"]
    with output_lock(out):
        check_or_write(out / "config.json", config)
        print(f"Preparing data on {device} with PyTorch {torch.__version__}.", flush=True)
        meta, ef_meta, _, data, _, sd, source_hash = prepare(
            out,
            config["data_root"],
            DEFAULTS["start_date"],
            DEFAULTS["end_date"],
            DEFAULTS["weather_map"],
        )
        digest = hashlib.sha256()
        for split in data:
            for value in split.data.values():
                digest.update(memoryview(value.contiguous().numpy()).cast("B"))
        check_or_write(
            out / "data_manifest.json",
            {
                "source_hash": source_hash,
                "data_hash": digest.hexdigest(),
                "counts": [len(split) for split in data],
                "device": str(device),
                "torch_version": str(torch.__version__),
            },
        )
        path = out / "results.json"
        results = read_json(path) if path.exists() else {}
        if config["reference_run"]:
            reused = reference_runs(
                Path(config["reference_run"]), out, config, data[2].data["label"].numpy()
            )
            if PROPOSED in results and results[PROPOSED] != reused:
                raise ValueError("Reference run changed. Choose a new output directory.")
            results[PROPOSED] = reused
            atomic_json(path, results)
            print(
                f"Verified and reused {len(reused)} Proposed-EF runs from the main benchmark.",
                flush=True,
            )
        for name, model_options in VARIANTS.items():
            runs = results.setdefault(name, [])
            completed = {record["seed"] for record in runs}
            for seed in seeds:
                if seed in completed:
                    continue
                print(f"{name} seed={seed} ({len(runs) + 1}/{len(seeds)})", flush=True)
                model, loaders, best, epoch = fit_nn(
                    PROPOSED,
                    params,
                    seed,
                    data,
                    config["final_epochs"],
                    config["final_patience"],
                    device,
                    meta,
                    ef_meta,
                    sd,
                    model_options=model_options,
                )
                test = metrics(*predict(model, loaders[2], PROPOSED, device))
                runs.append(
                    {
                        "seed": seed,
                        "best_epoch": epoch,
                        "val_RMSE": best,
                        "test": test,
                        "source": "ablation refit",
                    }
                )
                atomic_json(path, results)
                print(f"FINAL {name} seed={seed}: {test}", flush=True)
        export_figure(results, seeds, out)
        print(f"Finished. Figure: {out / 'ablation_distributions.png'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation-two-factor"))
    parser.add_argument("--params", type=Path, default=Path("results/reference/best_params.json"))
    parser.add_argument(
        "--reference-run", type=Path, help="Reuse a matching completed full-model run"
    )
    parser.add_argument("--seeds", type=int, choices=range(1, len(SEEDS) + 1), default=len(SEEDS))
    parser.add_argument("--final-epochs", type=int, default=DEFAULTS["final_epochs"])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    if args.final_epochs < 1:
        parser.error("--final-epochs must be positive")
    params = read_json(args.params)
    run(
        {
            "data_root": str(args.data_root.resolve()),
            "output": str(args.output.resolve()),
            "params": params.get(PROPOSED, params),
            "variants": VARIANTS,
            "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
            "final_epochs": args.final_epochs,
            "final_patience": 15,
            "max_len": 40,
            "selection_metric": "route_RMSE_kWh",
            "split_random_state": 42,
            "seeds_used": SEEDS[: args.seeds],
            "device": args.device,
        }
    )


if __name__ == "__main__":
    main()
