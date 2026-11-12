# Contributing

Install development dependencies with `python -m pip install -e ".[dev]"`.

Before proposing a change, run:

```bash
python -m pytest -q
python -m black --check src tests
```

Keep validation-based model selection separate from final test evaluation. Changes to features, route truncation, losses or split policy should be documented explicitly. Use synthetic fixtures in tests; do not commit real route records, IDs, predictions, weights, local paths or credentials.

Reference metrics describe the completed original experiment. Record new experiments in a separate output directory instead of replacing the reference results.
