# 电力负荷预测

本项目使用 XGBoost 完成小时级电力负荷预测。模型结合月份、小时、前 1～3 小时负荷以及昨日同时刻负荷，对历史负荷数据进行训练，并在测试集上评估预测效果。

## 主要功能

- 读取并清洗 `time`、`power_load` 格式的电力负荷数据
- 构造小时、月份 one-hot 特征
- 构造前 1 小时、前 2 小时、前 3 小时和昨日同时刻负荷特征
- 使用 XGBoost 训练回归模型
- 计算 MSE、RMSE、MAE 和 MAPE
- 保存模型到本地
- 模拟逐时刻预测，并绘制真实负荷与预测负荷对比图

## 项目结构

```text
load_predict_project/
├── data/
│   ├── fig/                     # 预测图等运行产物
│   ├── train.csv                # 训练数据
│   └── test.csv                 # 测试数据
├── log/                         # 运行日志
├── model/                       # 训练后的模型
├── src/
│   ├── train.py                 # 数据处理、特征工程、模型训练
│   └── predict.py               # 特征提取、预测和结果可视化
├── utils/
│   ├── common.py                # 数据预处理
│   └── log.py                   # 日志工具
├── .gitignore
└── README.md
```

## 数据说明

训练数据和测试数据均为 CSV 文件，必须包含以下两列：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `time` | 小时级时间，格式为 `YYYY/M/D H:MM` 或 `YYYY-MM-DD HH:MM:SS` | `2015/8/1 0:00` |
| `power_load` | 对应时刻的电力负荷 | `801.59` |

当前示例数据：

- `train.csv`：用于训练模型
- `test.csv`：从 2015 年 8 月 1 日开始进行测试预测

提交到 GitHub 前，请确认数据可以公开再分发。如果数据受隐私、版权或授权限制，请将 `data/*.csv` 加入 `.gitignore`，并在 README 中提供脱敏样例或数据获取说明。

## 环境要求

- Python 3.10 及以上
- pandas
- matplotlib
- xgboost
- scikit-learn 1.4 及以上
- joblib

创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas matplotlib xgboost scikit-learn joblib
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas matplotlib xgboost scikit-learn joblib
```

## 运行项目

脚本中的数据、模型和日志路径均以 `src` 为工作目录。首次运行前，需要把项目根目录加入 `PYTHONPATH`，以便正确导入 `utils` 包。

PowerShell：

```powershell
# 在项目根目录执行
$env:PYTHONPATH = (Resolve-Path .).Path
Set-Location .\src

# 训练并保存模型
python train.py

# 加载模型、执行预测并生成对比图
python predict.py
```

Linux 或 macOS：

```bash
# 在项目根目录执行
export PYTHONPATH="$(pwd)"
cd src

# 训练并保存模型
python train.py

# 加载模型、执行预测并生成对比图
python predict.py
```

训练完成后会生成：

```text
model/xgb_20261122.pkl
```

预测完成后会生成：

```text
data/fig/预测结果与实际结果对比分析.png
```

日志会写入 `log/` 目录。这些文件属于运行产物，默认不提交到 Git。

## 特征说明

模型使用以下特征：

- 24 个小时 one-hot 特征
- 12 个月份 one-hot 特征
- 前 1 小时负荷
- 前 2 小时负荷
- 前 3 小时负荷
- 昨日同时刻负荷

预测时只使用预测时刻以前的历史负荷，避免直接读取当前时刻的目标值。

## 输出指标

`train.py` 会输出：

- MSE：均方误差
- RMSE：均方根误差
- MAE：平均绝对误差
- MAPE：平均绝对百分比误差

`predict.py` 会输出测试时间段上的 MAE，并生成真实负荷与预测负荷的折线对比图。

## 注意事项

- 如果图表中的中文显示为方框，请安装中文字体，并修改脚本中的 `plt.rcParams["font.family"]`。
- `model/*.pkl` 是训练产物，默认不提交。克隆项目后需要先运行 `train.py`。
- 当前脚本使用固定的模型文件名 `xgb_20261122.pkl`。训练和预测必须使用同一个模型文件。
- 当前训练集与测试集采用随机划分。对时间序列任务进行更严格的评估时，建议改为按时间顺序划分，避免相邻时刻样本同时出现在训练集和测试集中。
