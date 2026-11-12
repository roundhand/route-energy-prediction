# Packaging validation

The extracted code was validated without rerunning the private full-budget experiment:

- Eight automated test cases passed, including all five neural architectures, shared labels/disjoint splits, sample-weighted route metrics, an all-eight-model synthetic CLI run, resume without repeated final seeds, and rejection of a changed configuration.
- Synthetic CSV inputs processed by the source notebook definitions and this package produced exactly equal dataset tensors in all three splits, for both energy-label and EF-label datasets.
- The source experiment's seed-1 checkpoints for DNN, FCNet, GRU, RET and Proposed-EF loaded strictly into both original and extracted architectures. CPU evaluation predictions matched exactly on the same synthetic inputs.
- Source and tests were formatted with Black. GitHub Actions is configured to run CPU tests and the formatting check; no hosted CI execution is claimed by this local validation.
- A Python wheel was built successfully from `pyproject.toml` with local build dependencies and without fetching runtime dependencies.

Tests verify extraction and execution, not full-scale statistical reproduction. The distributed reference metrics are the recorded original experiment. Original checkpoints used for the local migration audit are not distributed.
