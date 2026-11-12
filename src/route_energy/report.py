"""Write aggregate tables without exposing route IDs or predictions."""
import argparse
import json
from pathlib import Path
import pandas as pd


def export_results(results, out):
    rows = []
    lines = [
        "# Route energy prediction results",
        "",
        "Test metrics: mean ± population standard deviation across completed seeds.",
        "",
        "| Model | Seeds | MAE (kWh) | RMSE (kWh) | MAPE (%) | R2 | Trainable parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        runs = result["runs"]
        count = runs[0]["n_params"] if runs else None
        row = dict(
            model=name,
            n_seeds=len(runs),
            n_params=count,
            search_val_RMSE=result["search_val_RMSE"],
            **{k: v for k, v in result.items() if k.endswith("_mean") or k.endswith("_std")},
            best_params=json.dumps(result["best_params"], ensure_ascii=False),
        )
        if runs and all("train_seconds" in r for r in runs):
            row["refit_train_seconds_total"] = sum(r["train_seconds"] for r in runs)
            row["refit_train_seconds_mean"] = row["refit_train_seconds_total"] / len(runs)
            row["search_elapsed_seconds"] = result.get("search_elapsed_seconds")
        rows.append(row)
        formatted = [
            f"{result[k + '_mean']:.6f} ± {result[k + '_std']:.6f}"
            for k in ["MAE", "RMSE", "MAPE%", "R2"]
        ]
        lines.append(
            f"| {name} | {len(runs)} | "
            + " | ".join(formatted)
            + f" | {count if count is not None else 'N/A'} |"
        )
    lines += [
        "",
        "Distance regression is deterministic and is fitted once. Other reference models use 20 seeds.",
        "Tree model trainable parameter counts are not recorded. MAPE uses denominator y + 1e-8.",
        "",
    ]
    for name, result in results.items():
        lines += [
            "## " + name,
            "",
            "```json",
            json.dumps(result["best_params"], indent=2),
            "```",
            "",
        ]
    out = Path(out)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    export_results(
        json.loads((args.directory / "results.json").read_text(encoding="utf-8")), args.directory
    )


if __name__ == "__main__":
    main()
