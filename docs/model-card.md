# Pretrained model

| Property | Value |
|---|---|
| Model | NicheIR |
| Parameters | 2,917 |
| Training seed | 42 |
| Field-update steps | 48 |
| Query centre | Observed epithelial cell |
| Target / context radius | 72 / 216 µm |
| Classes | Immune, endothelial, stroma, epithelial |
| Development data | METABRIC and TONIC annotated breast cancer cells |
| Outputs | Four-class composition and expected missing counts |
| Weights | Bundled tensor-only PyTorch state dictionary |

NicheIR conditions reference intensity, reaction coefficients and symmetric directional coupling on visible cell counts and acquisition exposure. Forty-eight positive-field updates estimate a local intensity field. Integrating this field over missing target area gives expected missing counts, which are added to measured counts before normalization.

The pretrained weights were fitted with single-boundary masks. The same operator supports orthogonal-corner inputs. Composition is conditional on the four known lineages. Unknown cells contribute to the visible support hull and centroid, but not to lineage counts. The visible-cell hull is an observed-support proxy.

The weight checksum is stored in `src/nicheir/weights/model.json` and verified when loading. Preparation and inference leave the weights unchanged; user-trained checkpoints are saved separately. Outputs are point estimates, with exact empirical composition when the target is fully observed. Use independent reference data to evaluate a new tissue, assay or annotation vocabulary.
