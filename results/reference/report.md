# Route energy prediction results

Test metrics: mean ± population standard deviation across completed seeds.

| Model | Seeds | MAE (kWh) | RMSE (kWh) | MAPE (%) | R2 | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Distance-based Regression | 1 | 0.051970 ± 0.000000 | 0.082822 ± 0.000000 | 9.635385 ± 0.000000 | 0.973560 ± 0.000000 | 2 |
| XGBoost | 20 | 0.044831 ± 0.000122 | 0.071738 ± 0.000491 | 8.256373 ± 0.019613 | 0.980163 ± 0.000272 | N/A |
| LightGBM | 20 | 0.044659 ± 0.000090 | 0.071100 ± 0.000329 | 8.197319 ± 0.018536 | 0.980514 ± 0.000180 | N/A |
| DNN | 20 | 0.047629 ± 0.001305 | 0.072453 ± 0.001998 | 9.396270 ± 0.439483 | 0.979751 ± 0.001124 | 71809 |
| FCNet | 20 | 0.044308 ± 0.000861 | 0.067582 ± 0.001095 | 8.284137 ± 0.197912 | 0.982391 ± 0.000572 | 2433 |
| RET | 20 | 0.043848 ± 0.000688 | 0.067527 ± 0.000802 | 8.110498 ± 0.206884 | 0.982421 ± 0.000418 | 105409 |
| Proposed-EF | 20 | 0.041839 ± 0.000314 | 0.063777 ± 0.000697 | 7.820298 ± 0.099776 | 0.984320 ± 0.000345 | 272067 |

Distance regression is deterministic and is fitted once. Other reference models use 20 seeds.
Tree model trainable parameter counts are not recorded. MAPE uses denominator y + 1e-8.

## Distance-based Regression

```json
{
  "fit_intercept": true
}
```

## XGBoost

```json
{
  "n_estimators": 1000,
  "learning_rate": 0.037964144671773524,
  "max_depth": 4,
  "subsample": 0.8667261932357908,
  "colsample_bytree": 0.8117741317114515,
  "reg_alpha": 0.19386778259192655,
  "reg_lambda": 0.4758171862505622,
  "min_child_weight": 6.567848036397873
}
```

## LightGBM

```json
{
  "n_estimators": 600,
  "learning_rate": 0.030915236926517053,
  "max_depth": 10,
  "subsample": 0.6916766470490647,
  "colsample_bytree": 0.6963526277474572,
  "reg_alpha": 0.12980614096023002,
  "reg_lambda": 0.00031306364011981635,
  "num_leaves": 101,
  "min_child_samples": 20
}
```

## DNN

```json
{
  "lr": 0.00028180680291847244,
  "weight_decay": 2.4586032763280077e-06,
  "batch_size": 32,
  "hidden1": 256,
  "hidden2": 64,
  "dropout": 0.1800170052944527
}
```

## FCNet

```json
{
  "lr": 0.0029777723453189757,
  "weight_decay": 0.0001936445261894528,
  "batch_size": 64,
  "seg_loss_w": 0.3332931704983035
}
```

## RET

```json
{
  "lr": 0.00027010527749605503,
  "weight_decay": 2.9204338471814107e-05,
  "batch_size": 64,
  "seg_loss_w": 0.28140549728612524,
  "n_blocks": 2,
  "d_model": 64,
  "dropout": 0.06626289824631988
}
```

## Proposed-EF

```json
{
  "lr": 0.0005368463973198873,
  "weight_decay": 0.0007655080108692768,
  "batch_size": 64,
  "seg_loss_w": 0.21576320960238854,
  "d_model": 64,
  "n_heads": 2,
  "n_layers": 2,
  "d_ff": 512,
  "dropout": 0.05328574826366187
}
```
