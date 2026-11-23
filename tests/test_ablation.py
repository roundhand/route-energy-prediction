import json

import numpy as np
import pandas as pd
import pytest
import torch

from route_energy.ablation import METRICS, PROPOSED, VARIANTS, main
from route_energy.training import build


@pytest.mark.parametrize("name", list(VARIANTS)[1:])
def test_ablation_removes_only_targeted_components(prepared, name):
    _, _, (meta, ef_meta, _, ed, _, sd, _) = prepared
    p = dict(d_model=32, n_heads=2, n_layers=2, d_ff=64, dropout=0.1)
    model = build(PROPOSED, p, meta, ef_meta, sd, **VARIANTS[name]).eval()
    data = ed[0].data
    x, ctx, distance, mask, lengths = (
        data[key][:2] for key in ("seg_seq", "route_ctx", "cum_dist", "mask", "seg_lengths")
    )
    with torch.no_grad():
        segment, route = model(x, ctx, distance, mask, lengths)
        torch.testing.assert_close(route, (segment * lengths).sum(-1))
        assert torch.all(segment[~mask] == 0)
        if not VARIANTS[name].get("traffic_stability", True):
            changed = x.clone()
            changed[..., ef_meta.group_indices["traffic_stability"]] += 100
            torch.testing.assert_close(model(changed, ctx, distance, mask, lengths)[1], route)
        if not VARIANTS[name].get("distance_aware", True):
            assert model.dist_bias is None
            torch.testing.assert_close(model(x, ctx, distance + 1000, mask, lengths)[1], route)
        assert hasattr(model.seg_encoder, "group_attn")


def test_ablation_cli_export_and_resume(prepared, tmp_path):
    pytest.importorskip("matplotlib")
    raw, _, _ = prepared
    out = tmp_path / "ablation"
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            dict(
                d_model=16,
                n_heads=2,
                n_layers=2,
                d_ff=32,
                dropout=0.05,
                lr=5e-4,
                weight_decay=7e-4,
                batch_size=16,
                seg_loss_w=0.2,
            )
        )
    )
    args = [
        "--data-root",
        str(raw),
        "--output",
        str(out),
        "--params",
        str(params),
        "--seeds",
        "2",
        "--final-epochs",
        "1",
        "--device",
        "cpu",
    ]
    main(args)
    results_path = out / "results.json"
    before = results_path.read_bytes()
    results = json.loads(before)
    scores = pd.read_csv(out / "ablation_per_seed.csv")
    assert scores["Model"].drop_duplicates().tolist() == [
        PROPOSED,
        "w/o Traffic Stability",
        "w/o Distance-Aware",
        "w/o Both",
    ]
    for name, runs in results.items():
        rows = scores[scores["Model"] == name]
        assert rows["Seed"].tolist() == [1, 7]
        for key, label in METRICS.items():
            scale = 1 if key == "MAPE%" else 1000
            np.testing.assert_allclose(rows[label], [run["test"][key] * scale for run in runs])
    assert (out / "ablation_distributions.png").is_file()
    assert (out / "ablation_distributions.svg").is_file()
    main(args)
    assert results_path.read_bytes() == before
