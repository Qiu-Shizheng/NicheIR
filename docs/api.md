# Python API

```python
from nicheir import (
    Cells, HalfPlane, Corner, PreparedBatch,
    prepare_cells, read_cells_csv, load_inputs, save_inputs,
    validate_inputs, load_pretrained, Predictor, Prediction,
)
```

## Preparation

- `Cells(cell_id, xy, class_id)`: arrays of IDs, `[N,2]` micrometre coordinates and class IDs −1/0/1/2/3. All rows refer to a single visible field.
- `read_cells_csv(path) -> Cells`: reads `cell_id,x,y,class_id`, ignoring additional columns.
- `HalfPlane(cut: float, angle: float = 0)`: an outward-normal angle in radians and a canonical first-axis cut in micrometres.
- `Corner(cut_x: float, cut_y: float, orientation: int = 0)`: orthogonal cuts and a quarter-turn orientation 0–3.
- `prepare_cells(cells, query_ids, acquisition) -> PreparedBatch`: produces validated arrays and retains the query order. Unsupported inputs raise `ValueError`.
- `PreparedBatch.inputs`: the six-array dictionary; `.query_ids`: ordered IDs; `.save(path)`: NPZ; `.save_query_index(path)`: CSV mapping window/slot to query ID.
- `validate_inputs(mapping)`: validates and returns independent float32/bool arrays.
- `load_inputs(path)` and `save_inputs(path, mapping)`: validated NPZ I/O without object arrays or pickle.

## Pretrained inference

`load_pretrained(device="cpu") -> Predictor` loads the included tensor-only state dictionary, checks its SHA256 and the 2,917-parameter architecture, and enters evaluation mode. Initial construction uses seed 42 without changing the caller's CPU RNG state. No network operation occurs.

`Predictor.predict(inputs, batch_windows=4) -> Prediction` validates the six inputs and processes bounded window batches. `Predictor.predict_npz(path, output=None, batch_windows=4)` loads a supported archive and can directly save the result. `batch_windows` must be positive. Optional devices follow PyTorch device syntax.

`Prediction` contains NumPy arrays named `composition`, `missing_counts`, `missing_counts_by_bin`, `field_density` and `valid`. `.save(path)` writes these arrays to NPZ. The class order is available as `predictor.class_names` or `nicheir.CLASS_NAMES`.

The Python API leaves PyTorch thread settings unchanged. The command line uses two CPU threads by default. Inference is deterministic within a fixed software/device environment; different numerical libraries or devices can yield small floating-point differences.

## Training and checkpoints

- `initialize(seed=42, device="cpu") -> Predictor`: untrained instance of the same architecture.
- `load_checkpoint(path, device="cpu") -> Predictor`: loads a tensor-only checkpoint and validates fixed geometry buffers.
- `Predictor.save_checkpoint(path)`: saves CPU tensors without inputs, optimizer state or user metadata.
- `TrainingConfig(...)`: epoch count, batch size, optimizer settings, seed and objective weights.
- `fit(inputs, hidden_counts, config=None, initialization="pretrained", device="cpu", window_weights=None) -> TrainingResult`: fits for the full requested epoch count.
- `validate_targets(inputs, hidden_counts)`: checks the separate `[W,Q,2,16,4]` supervised labels.
- `TrainingResult.predictor`, `.history`, `.config`, `.initialization`: final model and numeric fit record; `.save("adapted.pt")` writes weights and a JSON history.

See [the training tutorial](training.md) for the objective, label construction and weighting.

