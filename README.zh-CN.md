# 路径能耗预测与统一调参实验

这是本次训练流程的独立代码版本，可作为 GitHub 仓库上传。模型定义、数据处理和训练入口已从原 notebook 拆出，运行时不读取 notebook。

## 安装与运行

建议使用 Python 3.9–3.12，并在项目根目录执行：

```bash
python -m pip install -e ".[dev]"
```

如需 GPU，先安装适合本机的 PyTorch 2.5 CUDA 版本。本次历史实验使用 Python 3.9、PyTorch 2.5.1+cu121；`requirements-reference.txt` 记录历史环境，不能保证适用于所有新 Python 版本。

先用完全合成的数据检查流程：

```bash
python -m route_energy.synthetic --output data/demo
python -m route_energy --data-root data/demo --output outputs/demo --models "FCNet" "Proposed-EF" --trials 1 --search-epochs 1 --final-epochs 1 --seeds 1 --device cpu
python -m pytest -q
```

合成数据只用于运行检查，不用于验证论文精度。真实数据应按 [CSV 格式说明](docs/data-format.md) 组织：

```bash
python -m route_energy --config configs/experiment.json --data-root "D:/datasets/ev_energy"
```

配置文件中的路径相对于执行命令时的目录解析。更换实验日期时，同时修改日期范围和 `weather_map`。

## 实验设置

- 模型：距离线性回归、XGBoost、LightGBM、DNN、FCNet、GRU、RET、Proposed-EF。
- 所有模型使用固定路径 ID 划分，训练/验证/测试约为 80%/10%/10%，划分种子为 42。
- FCNet、GRU、RET 保持路径 MSE + 可调权重 × 路段能耗 MSE；Proposed-EF 的路段项为 EF MSE。DNN 只有路径输出。
- 调参、神经网络早停、剪枝和最佳权重选择，统一使用整个验证集的路径级 RMSE（kWh）。测试集仅用于最终评估。
- 默认每个随机模型 30 次搜索，最多 50 epochs、patience=10；确定最佳参数后，用原来的 20 个种子复训，最多 100 epochs、patience=15。
- 距离回归穷举是否拟合截距两个选项，确定性结果只记录一次。
- 每条路径最多保留前 40 个路段，评价标签也是这些路段对应的能耗之和。

完整预算耗时较长。断点续跑时执行同一命令即可，已经完成的 trial 与最终种子会被保留。不要同时向同一输出目录运行两个任务。修改配置或源码后使用新输出目录；数据文件也必须保持不变。强制终止可能留下 `.run.lock`，确认进程已停止后再移除。

## 消融实验与分布图

```bash
python -m pip install -e ".[ablation]"
python -m route_energy.ablation --data-root "D:/datasets/ev_energy" --device cuda
```

默认读取 `results/reference/best_params.json` 的新版 Proposed-EF 参数，并使用主实验的 20 个种子、固定划分和验证路径 RMSE 早停。`w/o Traffic Stability` 去除交通稳定性特征组。`w/o Distance-Aware` 去除距离位置编码和注意力距离偏置，保留学习式顺序嵌入及 FiLM。`w/o Both` 同时进行这两项消融。四组模型均使用分组编码器。

如果已有匹配的主实验输出，可追加 `--reference-run <主实验输出目录>`。该选项核对超参数、各划分的行程 ID、种子、测试标签与指标后，复用完整模型的逐种子结果。每个种子结束后写入 `outputs/ablation-two-factor/results.json`，使用同一命令可续跑。全部完成后输出 `ablation_distributions.png`、`ablation_distributions.svg`、图注和两份 CSV。图与 CSV 的 MAE、RMSE 使用 Wh，MAPE 使用 %。

## 文件与结果

| 路径 | 作用 |
|---|---|
| `src/route_energy/` | 独立训练代码，模型、数据、搜索、训练、评价分别管理 |
| `configs/experiment.json` | 本次实验默认预算、日期、天气映射 |
| `docs/data-format.md` | 数据目录、CSV 列名、单位与类别说明 |
| `docs/reproducibility.md` | 搜索范围、来源、复现边界 |
| `results/reference/` | 本次已完成实验的最佳参数、每种子指标和汇总报告 |
| `tests/` | 合成数据、模型聚合、最佳 checkpoint 与断点续跑检查 |
| `outputs/` | 新运行结果，默认不会提交到 Git |

参考结果见 [report.md](results/reference/report.md)。该目录不包含原始数据、行程 ID、逐行预测、权重、原日志或数据库，因此别人需要自行获得相同数据才能复现相同数值。

新运行会额外记录每个种子的参数量、训练耗时和包含最终评估的耗时。训练耗时包含模型初始化、训练和每轮验证；不等于纯 GPU 计算时长。历史实验缺失的复训耗时没有补造。树模型参数量仍记为空。

本目录默认使用 MIT 代码许可证，见 [LICENSE](LICENSE)。原始数据的使用和分发权限单独处理。上传 GitHub 时选择这个项目目录；数据、权重和运行输出已经加入 `.gitignore`。
