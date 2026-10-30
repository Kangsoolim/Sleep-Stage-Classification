# Sleep-EDF 睡眠 EEG 五分类课程项目

本项目用于《生物医学传感检测与仪器》课程：先复现 MNE-Python 的睡眠分期教程，再用相同的 EEG 频带功率特征比较 Standard Random Forest 与 `class_weight="balanced"` Random Forest。主实验按受试者划分训练和测试数据。

## 数据和方法来源

- 数据：[PhysioNet Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) 中的 Sleep Cassette；版本 DOI：[10.13026/C2X676](https://doi.org/10.13026/C2X676)。使用 MNE `sleep_physionet.age.fetch_data()` 按需获取 EDF 和 Hypnogram。
- 官方方法：[MNE-Python Sleep stage classification from polysomnography (PSG) data](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)；本仓库的复现位于 `notebooks/01_mne_official_reproduction.ipynb`。

仅使用 Fpz-Cz、Pz-Oz 两路 EEG。将 Sleep stage 3/4 合并为 N3，形成 W、N1、N2、N3、REM 五类；保留睡眠前后各约 30 分钟 Wake，按 30 秒切 epoch。特征是 0.5–30 Hz 归一化 PSD 中 delta、theta、alpha、sigma、beta 五个频带在两路 EEG 上的相对功率，共 10 维；具体为归一化 PSD 的频带内均值，而非频带积分。0.5–30 Hz 是 PSD 分析范围，不是时域滤波步骤。

## 环境与安装

已验证环境为 Python 3.12.14；依赖的精确版本见 `requirements.txt`（MNE 1.13.2、scikit-learn 1.9.1 等）。在项目根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import mne, sklearn; print(mne.__version__, sklearn.__version__)"
```

首次执行需要能访问 PhysioNet 或其公开镜像，并留出本地 EDF 缓存空间。后续执行会复用 `data/mne_data/`，无需删除缓存重新下载。

## 运行官方复现

在项目根目录运行以下命令；`nbconvert --execute` 会启动新 kernel，执行输出保存在被 Git 忽略的 `tmp/`：

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
.\.venv\Scripts\python.exe -m jupyter nbconvert --execute --to notebook notebooks\01_mne_official_reproduction.ipynb --output official_executed.ipynb --output-dir tmp
```

官方复现使用受试者 0 的第 1 晚训练、受试者 1 的第 1 晚测试，输出见 `results/reproduction/` 和 `figures/reproduction/`。本地 Windows 受限环境如无法在用户目录创建 Jupyter 运行文件，可将 `IPYTHONDIR`、`JUPYTER_CONFIG_DIR`、`JUPYTER_DATA_DIR`、`JUPYTER_RUNTIME_DIR` 指向 `tmp/` 下的目录。

## 运行主实验与图表

以下命令按顺序执行。第一步从已有 EDF 缓存重新生成 epoch 特征及固定的受试者分组；第二步在保存的同一组 folds 上训练两种 RF 并写出 OOF 结果；第三步从结果 CSV 生成图表和评估表格。

```powershell
.\.venv\Scripts\python.exe src\build_main_experiment_data.py
.\.venv\Scripts\python.exe src\run_grouped_rf_experiments.py
.\.venv\Scripts\python.exe src\build_evaluation_figures.py
```

主实验固定为受试者 0–9、每人第 1 晚、`GroupKFold(n_splits=5, groups=subject_id)`。每折训练和测试的受试者不重叠；Standard 和 Balanced RF 使用完全相同的 folds、特征和 `n_estimators=100, random_state=241`，唯一模型差异是 `class_weight`。每个 epoch 恰好产生一次样本外预测。`notebooks/02_main_experiments.ipynb` 保留设计记录；上述 `src/` 脚本是完整主实验入口。数据准备、训练及图表命令会写入固定结果路径；重新运行前请自行备份需要保留的结果。

## 主要结果

主实验共 9,913 个 30 秒 epoch。以下为五折汇总的 OOF 指标，精确值以 `results/overall_metrics.csv` 为准：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Standard RF | 0.7689 | 0.6894 |
| Balanced RF | 0.7664 | 0.7061 |

Balanced RF 的 N1 F1 从 0.2980 变为 0.3565，REM F1 从 0.6933 变为 0.7047；Accuracy 略降。逐折、逐类、OOF 预测与混淆矩阵分别见 `results/fold_metrics.csv`、`results/per_class_metrics.csv`、`results/oof_predictions.csv`、`results/confusion_matrices.csv`。

## Exploratory spectral entropy extension

Core Experiment 保持不变。Spectral entropy 是独立的附加特征实验：在两个 EEG 通道各追加一个 0.5–30 Hz normalized spectral entropy，将 Balanced RF 输入从 10D 扩为 12D；使用原 accepted cache 和完全相同的五折 subject-wise GroupKFold。首次复现时，先按上一节顺序完成数据准备和 10D 主实验，再执行扩展脚本。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe src\run_spectral_entropy_extension.py
```

输出写入 `results/spectral_entropy_extension/`，Table 4 写入 `results/table_spectral_entropy_extension.csv`，Figure 6 写入 `figures/figure_6_spectral_entropy_extension.png`。本次 12D 结果为 Accuracy 0.7701、Macro-F1 0.7086，相对 10D 分别增加 0.0037 和 0.0024；变化幅度较小且各阶段方向不完全一致，详见 extension summary 与逐类 CSV。大型 12D cache 保持在被 `.gitignore` 排除的 `data/processed/`。

## 目录与数据入库约定

| 目录 | 内容 |
|---|---|
| `notebooks/` | 官方教程复现 notebook、主实验设计记录 |
| `src/` | 数据准备、特征、分组训练和图表生成代码 |
| `results/` | 小型、可审计的 CSV/JSON 与结果摘要 |
| `figures/` | 可复现 PNG 图和评估图 PDF |
| `data/` | 数据说明；EDF 和大型特征缓存仅保存在本地 |

原始 `*.edf`、MNE 缓存、`data/processed/` 大型中间特征、虚拟环境与临时文件均不进入 Git。小型结果 CSV、图表与正式 notebook 保留在仓库，便于独立核对实验数值。


## 数据许可与引用

Sleep-EDF Expanded v1.0.0 按 [Open Data Commons Attribution License v1.0](https://physionet.org/content/sleep-edfx/1.0.0/) 提供。数据仅通过官方来源按需下载，不随本仓库分发；使用数据或复用相关结果时，请遵守数据许可，并按官方页面要求保留数据来源和引用。

- 数据版本：PhysioNet Sleep-EDF Database Expanded v1.0.0，DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676)。
- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on Biomedical Engineering. 2000;47(9):1185–1194. [原始论文](https://ieeexplore.ieee.org/document/867928/)
- Mourtazaev MS, Kemp B, Zwinderman AH, Kamphuisen HAC. Age and gender affect different characteristics of slow waves in the sleep EEG. Sleep. 1995;18(7):557–564. [数据集参考文献](https://physionet.org/content/sleep-edfx/1.0.0/#references)
- 方法复现参考 [MNE-Python 官方睡眠分期教程](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)，并请按数据集页面要求引用 PhysioNet。

上述许可是数据集的许可，不代表已为本仓库全部代码授予同一许可。

结果元数据中的文件路径均相对于项目根目录保存；脚本会在运行时解析本地缓存路径，无需使用原开发电脑的目录。
