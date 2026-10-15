# Train or fine-tune NicheIR

Fit or fine-tune NicheIR on prepared visible inputs and separate hidden-count labels. Assign patient-disjoint training and evaluation sets before fitting.

## Try a complete synthetic example

```bash
nicheir train-example --output-dir training_example --epochs 2
nicheir predict --inputs training_example/inputs.npz \
  --checkpoint training_example/trained.pt --output trained_predictions.npz
```

The example generates a complete synthetic cell cloud, then uses only visible cells to build the six model inputs. Hidden cells supply separate supervised labels. It performs two epochs, saves the final weights, reloads them and verifies that predictions are identical. 

## Prepare supervised targets

The input NPZ follows the same [six-array contract](input-contract.md) as inference. A separate target NPZ contains exactly one array:

| Key | Shape | Meaning |
|---|---|---|
| `hidden_counts` | `[W,Q,2,16,4]` | Nonnegative integer counts of hidden cells in the inner two polar rings, by sector and class |

Construct these labels from complete reference neighbourhoods using the **same centre exclusion, ring boundaries and sector assignment** as preprocessing. Subtract observed counts from complete counts per bin. Keep hidden cells and reference labels in the target arrays only.

Targets must be zero in fully observed bins and padded slots. Positive hidden counts at zero predicted Poisson rate are an error, not replaced by an epsilon rate. The training API rejects noninteger, nonfinite, negative, wrongly shaped or geometrically incompatible labels.

## Command line

```bash
nicheir train --inputs train_inputs.npz --targets train_targets.npz \
  --output adapted.pt --epochs 20 --batch-windows 4 --seed 42
nicheir predict --inputs evaluation_inputs.npz --checkpoint adapted.pt \
  --output evaluation_predictions.npz
```

The default initialization uses bundled pretrained weights. Add `--initialization random` to train a new instance with seed 42. `--device cpu` is the default; supported PyTorch devices may be supplied explicitly. No bundled weight file is overwritten.

`adapted.pt` is a tensor-only state dictionary. `adapted.json` records the numeric configuration and epoch history. The checkpoint is the final epoch, not a checkpoint selected on an evaluation set. Optimizer state is not saved, so exact interrupted-run continuation is not provided.

## Python API and weighting

```python
import numpy as np
from nicheir import TrainingConfig, fit, load_inputs

inputs = load_inputs("train_inputs.npz")
with np.load("train_targets.npz", allow_pickle=False) as archive:
    labels = archive["hidden_counts"]
result = fit(inputs, labels, config=TrainingConfig(epochs=20, seed=42))
result.save("adapted.pt")
print(result.history[-1])
```

`fit(..., window_weights=weights)` accepts optional positive `[W]` training weights. Weights are normalized once to mean one. Without weights, each window has equal weight and valid queries within each window have equal weight. Choose weights using training metadata if patients contribute unequal numbers of fields. The loop runs the requested epochs and returns the final model.

The objective per window is local composition total variation (TV), plus `0.5 ×` TV of the mean query composition, `0.1 ×` a dominant-type margin loss and `0.05 ×` the proper hidden-bin Poisson negative log likelihood. The Poisson term includes log-factorials and is divided by 128 (two rings × 16 sectors × four types). Empty complete neighbourhoods use a uniform composition. AdamW defaults to learning rate 0.0003, weight decay 0.0001 and gradient-norm clipping at 5. The learning rate is constant, and one epoch visits all supplied windows in a shuffled order. `TrainingConfig` exposes these controls.

Training randomness uses a local seed, default 42. Repeated CPU fits are tested for identical weights in the same environment. Maintain separate evaluation data and report the exact training settings when applying the package to a new cohort.
