import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from route_energy.metrics import metrics
from route_energy.training import build, fit_nn, forward, predict
from route_energy.experiment import main, MODELS


def test_shared_labels_and_disjoint_splits(prepared):
    _, out, (_, _, baseline, ef, _, _, _) = prepared
    sets = [
        set(map(tuple, pd.read_csv(out / f"{s}_ids.csv").values)) for s in ["train", "val", "test"]
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for b, e in zip(baseline, ef):
        torch.testing.assert_close(b.data["label"], e.data["label"])
        torch.testing.assert_close(
            e.data["label"], (e.data["seg_labels"] * e.data["seg_lengths"]).sum(-1)
        )
        assert torch.all(b.data["seg_seq"][~b.data["mask"]] == 0)


def test_metrics_use_all_samples():
    result = metrics([0, 2, 3], [1, 2, 5])
    assert result["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert result["MAE"] == 1


@pytest.mark.parametrize("name", ["DNN", "FCNet", "GRU", "RET", "Proposed-EF"])
def test_reference_counts_aggregation_and_best_checkpoint(prepared, name):
    _, _, (meta, ef_meta, bd, ed, _, sd, _) = prepared
    reference = json.loads(
        (Path(__file__).parents[1] / "results/reference/results.json").read_text(encoding="utf-8")
    )
    params = dict(reference[name]["best_params"])
    model = build(name, params, meta, ef_meta, sd)
    assert (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        == reference[name]["runs"][0]["n_params"]
    )
    if name != "DNN":
        batch = {k: v[:2] for k, v in (ed if name == "Proposed-EF" else bd)[0].data.items()}
        seg, route, _, _, mask = forward(model, batch, name, "cpu")
        expected = (seg * batch["seg_lengths"]).sum(-1) if name == "Proposed-EF" else seg.sum(-1)
        torch.testing.assert_close(route, expected)
        assert torch.all(seg[~mask] == 0)
    data = (
        [TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) for x, y in sd]
        if name == "DNN"
        else ed
        if name == "Proposed-EF"
        else bd
    )
    trained, loaders, best, epoch = fit_nn(
        name, params, 42, data, 2, 2, torch.device("cpu"), meta, ef_meta, sd
    )
    y, predictions = predict(trained, loaders[1], name, "cpu")
    assert metrics(y, predictions)["RMSE"] == pytest.approx(best)
    assert epoch in [1, 2]


def test_all_models_cli_and_resume(prepared, tmp_path):
    raw, _, _ = prepared
    out = tmp_path / "run"
    args = [
        "--data-root",
        str(raw),
        "--output",
        str(out),
        "--trials",
        "1",
        "--search-epochs",
        "1",
        "--final-epochs",
        "1",
        "--seeds",
        "1",
        "--device",
        "cpu",
    ]
    main(args)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert set(results) == set(MODELS)
    for name, result in results.items():
        assert len(result["runs"]) == 1
        run = result["runs"][0]
        assert 0 < run["train_seconds"] <= run["run_seconds"]
        assert np.isfinite(run["test"]["RMSE"])
        assert (out / (name.replace(" ", "_") + "_trials.csv")).exists()
    before = (out / "results.json").read_bytes()
    main(args)
    assert before == (out / "results.json").read_bytes()
    assert not (out / ".run.lock").exists()
    assert (out / "report.md").exists()
    changed = list(args)
    changed[changed.index("--trials") + 1] = "2"
    with pytest.raises(ValueError, match="configuration changed"):
        main(changed)
