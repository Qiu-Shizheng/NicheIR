# NicheIR

**Reconstruct local cell-type composition at spatial acquisition boundaries.**

NicheIR uses annotated, observed cells and a known acquisition boundary to estimate immune, endothelial, stromal and epithelial composition around visible epithelial cells. Its 2,917-parameter spatial count operator combines acquisition exposure, local tissue context and directional coupling. Estimated missing counts are added to observed counts to recover the composition of a 72 µm neighbourhood.

The package includes pretrained weights, coordinate preprocessing, a Python API, a command-line interface and supervised training. CPU inference works offline.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install "git+https://github.com/Qiu-Shizheng/NicheIR.git@v0.1.1"
```

The [current release](https://github.com/Qiu-Shizheng/NicheIR/releases/tag/v0.1.1) also provides a wheel, source distribution and [source ZIP](https://github.com/Qiu-Shizheng/NicheIR/releases/download/v0.1.1/NicheIR_GitHub_Ready.zip). With NumPy, SciPy and PyTorch already installed, the wheel can be installed offline:

```bash
python -m pip install --no-index --no-deps nicheir-0.1.1-py3-none-any.whl
```

## Quick start

```bash
nicheir example --geometry halfplane --output-dir example_run/halfplane
nicheir example --geometry corner --output-dir example_run/corner
```

Each example generates synthetic cells with seed 42, prepares model inputs and saves predictions. Read the four-class compositions as follows:

```python
import numpy as np

with np.load("example_run/corner/predictions.npz") as output:
    compositions = output["composition"][output["valid"]]
    print(compositions)  # immune, endothelial, stroma, epithelial
```

## Use your cells

Supply one visible acquisition field per CSV, with coordinates in **micrometres**:

```csv
cell_id,x,y,class_id
cell_a,-25.0,-18.0,3
cell_b,-18.0,-24.0,0
cell_c,-40.0,-11.0,2
cell_d,-45.0,-38.0,-1
```

| class_id | Lineage |
|---|---|
| 0 | Immune |
| 1 | Endothelial |
| 2 | Stroma |
| 3 | Epithelial |
| -1 | Unknown; used for visible support, excluded from lineage counts |

Create `queries.csv` with a `cell_id` column identifying observed epithelial cells. Include all observed cells from the field in `cells.csv`; the query cell is excluded from its own neighbour counts.

For a field visible at `x <= 0`:

```bash
nicheir prepare --cells cells.csv --queries queries.csv \
  --geometry halfplane --cut 0 --angle 0 \
  --output inputs.npz --query-index query_index.csv
nicheir predict --inputs inputs.npz --output predictions.npz
```

For a corner visible at `x <= 0 AND y <= 0`, use `--geometry corner --cut 0 --second-cut 0 --orientation 0`. Both query-to-edge distances must be below 72 µm. Halfplane angles are outward-normal angles in radians; corner orientations are quarter turns. The declared geometry must describe visibility throughout the 216 µm context. See [preprocessing](docs/preprocessing.md) for supported orientations and boundary cases.

## Python API

```python
from nicheir import HalfPlane, load_pretrained, prepare_cells, read_cells_csv

cells = read_cells_csv("cells.csv")
batch = prepare_cells(cells, ["cell_a"], HalfPlane(cut=0.0))
prediction = load_pretrained().predict(batch.inputs)
print(prediction.composition[prediction.valid])
prediction.save("predictions.npz")
```

For prepared arrays, use `load_pretrained().predict_npz("inputs.npz")`. The [input contract](docs/input-contract.md) defines the six arrays and output shapes; the [API reference](docs/api.md) covers batching and checkpoints.

## Train or fine-tune

```bash
nicheir train-example --output-dir training_example --epochs 2
nicheir train --inputs train_inputs.npz --targets train_targets.npz \
  --output adapted.pt --epochs 20 --seed 42
```

Supervised labels are separate hidden-bin counts. Training defaults to the bundled initialization and saves the final epoch; random initialization is also available. Prepare patient-disjoint training and evaluation sets before fitting. See the [training tutorial](docs/training.md) for targets, objective and weighting.

## Documentation

- [中文快速上手](docs/quickstart.zh.md)
- [Coordinate preprocessing](docs/preprocessing.md)
- [Input and output contract](docs/input-contract.md)
- [Python API](docs/api.md)
- [Training](docs/training.md)
- [Pretrained model](docs/model-card.md)

The input annotations and acquisition geometry are supplied by the user. The pretrained model targets epithelial-centred neighbourhoods and the four supported lineages; its weights were developed on METABRIC and TONIC breast cancer data.

## Development

```bash
python -m pip install -e ".[test,build]"
python -m pytest
python -m build
```

An optional [GitHub Actions template](ci/github-actions.yml) runs tests and package examples.

## Licence

Code and pretrained weights are available under the [MIT License](LICENSE).
