# 本地数据说明

数据为 [PhysioNet Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) 的 Sleep Cassette 子集，版本 DOI 为 [10.13026/C2X676](https://doi.org/10.13026/C2X676)。通过 MNE `sleep_physionet.age.fetch_data()` 获取 PSG EDF 和配套 Hypnogram；官方复现使用受试者 0、1 的第 1 晚，主实验使用受试者 0–9 的第 1 晚。

本机 EDF 缓存位于 `data/mne_data/`，大型 epoch 特征位于 `data/processed/`；两者均不进入 Git。已获取文件的相对名称见 `results/reproduction/fetched_files.json` 和 `results/main_experiment/fetched_files.json`，主实验样本数及标签分布见 `results/main_experiment/data_summary.json`。保留本地缓存即可重复运行，无需强制重新下载。


结果 JSON 中的缓存和文件路径相对于项目根目录保存。迁移项目或首次克隆后，先执行 README 中的数据准备命令，由 MNE 获取数据并建立本机缓存。

数据许可为 Open Data Commons Attribution License v1.0；许可、原始研究论文及 PhysioNet 引用要求见 [数据集官方页面](https://physionet.org/content/sleep-edfx/1.0.0/) 和项目 README 的“数据许可与引用”。
