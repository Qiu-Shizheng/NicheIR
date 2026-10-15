# 中文快速上手

NicheIR 根据**已经观测并注释的细胞坐标**及**已知采集边界**，估计可见上皮细胞周围 72 µm 圆形邻域的四类组成。类别顺序固定为免疫、内皮、基质、上皮。包内自带模型权重，可在 CPU 离线运行。

## 安装与合成示例

在下载的源码目录中运行：

```bash
python -m pip install .
nicheir example --geometry corner --output-dir example_run
```

示例使用固定 seed 42 生成合成细胞，不包含患者数据。输出有 `cells.csv`、`queries.csv`、`inputs.npz`、`query_index.csv` 和 `predictions.npz`。用 `valid=True` 选择有效结果。

## 自己的数据

CSV 必须是**一个可见视野**，包含 `cell_id,x,y,class_id`；坐标单位为微米。类型映射为 0=免疫，1=内皮，2=基质，3=上皮，−1=未知。未知细胞参与可见支持域计算，但不进入四类计数。查询 CSV 仅需 `cell_id` 列，每个查询必须是可见上皮细胞。

```bash
nicheir prepare --cells cells.csv --queries queries.csv \
  --geometry halfplane --cut 0 --angle 0 \
  --output inputs.npz --query-index query_index.csv
nicheir predict --inputs inputs.npz --output predictions.npz
```

以上表示可见区域 `x <= 0`。角点使用 `--geometry corner --cut 0 --second-cut 0 --orientation 0`，表示 `x <= 0 且 y <= 0`。角点只支持四个已定义轴向，查询距两条边界都必须小于 72 µm。

输入前先完成细胞注释、坐标尺度核实和采集边界定义。零曝光格中存在观测细胞时会报错，不会自动修改坐标或删除查询。更详细的几何、输入和输出定义见 [preprocessing.md](preprocessing.md) 与 [input-contract.md](input-contract.md)。

## 训练与微调

`nicheir train-example --output-dir training_example --epochs 2` 可用合成细胞完成训练、保存及重新加载验证。自己的训练数据需要六数组输入及独立的 `hidden_counts[W,Q,2,16,4]` 标签。默认 seed42、预训练初始化、固定轮数；也支持随机初始化。详见[训练教程](training.md)。训练前请按患者划分训练集与评估集。

## GitHub 安装与离线包

可直接安装固定版本：

```bash
python -m pip install "git+https://github.com/Qiu-Shizheng/NicheIR.git@v0.1.1"
```

也可从 [v0.1.1 发行页](https://github.com/Qiu-Shizheng/NicheIR/releases/tag/v0.1.1) 下载 ZIP 或 wheel。离线安装前须备好 NumPy、SciPy 和 PyTorch；模型权重已包含在包内。
