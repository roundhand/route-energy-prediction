"""Independent per-model search spaces from the original experiment."""


def suggest(trial, name):
    if name == "Distance-based Regression":
        return {"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}
    if name in ["XGBoost", "LightGBM"]:
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-05, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-05, 10.0, log=True),
        )
        if name == "LightGBM":
            p["num_leaves"] = trial.suggest_int("num_leaves", 8, min(128, 2 ** p["max_depth"]))
            p["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 80)
        else:
            p["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True)
        return p
    p = dict(
        lr=trial.suggest_float("lr", 0.0001, 0.003, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-06, 0.01, log=True),
    )
    if name != "Proposed-EF":
        p["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    else:
        p["batch_size"] = 64
    if name != "DNN":
        p["seg_loss_w"] = trial.suggest_float("seg_loss_w", 0.05, 0.5)
    if name == "DNN":
        p.update(
            hidden1=trial.suggest_categorical("hidden1", [64, 128, 256]),
            hidden2=trial.suggest_categorical("hidden2", [32, 64, 128]),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
        )
    if name == "RET":
        p.update(
            n_blocks=trial.suggest_int("n_blocks", 1, 3),
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
        )
    if name == "Proposed-EF":
        p.update(
            d_model=trial.suggest_categorical("d_model", [64, 128, 256]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            n_layers=trial.suggest_int("n_layers", 2, 4),
            d_ff=trial.suggest_categorical("d_ff", [128, 256, 512]),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
        )
    return p
