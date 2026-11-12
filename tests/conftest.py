import pytest
import torch
from route_energy.data import prepare
from route_energy.experiment import DEFAULTS
from route_energy.synthetic import generate


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    torch.set_num_threads(2)
    base = tmp_path_factory.mktemp("synthetic")
    raw = generate(base / "raw", trips_per_day=10)
    out = base / "prepared"
    out.mkdir()
    data = prepare(out, raw, "2023-09-11", "2023-09-15", DEFAULTS["weather_map"])
    return raw, out, data
