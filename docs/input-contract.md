# Prepared input contract

An input NPZ contains **exactly** the following six arrays. It must not contain targets, complete-reference counts or other additional arrays. `W` is the number of windows and `Q` is 1–32 slots. Each window has at least one valid slot. The cell-CSV preparer always pads to 32 slots and provides a separate query index.

| Array | Shape | Type and meaning |
|---|---|---|
| `polar_counts` | `[W,Q,4,16,4]` | float32 nonnegative integer observed-cell counts, indexed by ring, sector, class |
| `crop_coverage` | `[W,Q,4,16]` | float32 acquisition-area fraction of each polar bin |
| `visible_hull_support` | `[W,Q,4,16]` | float32 bin-area fraction inside the convex hull of all supplied visible cells |
| `local_observed_counts` | `[W,Q,4]` | float32 sum of the first two count rings |
| `query_features` | `[W,Q,7]` | float32 features described below |
| `valid` | `[W,Q]` | boolean valid-query indicator; numeric 0/1 is also accepted and converted |

All values must be finite. Coverage and support lie in [0,1], support cannot exceed coverage beyond a 2×10⁻⁶ numerical tolerance, and **zero exposure cannot contain any observed count**. Counts must be exactly representable as float32 integers. Invalid slots have zeros in every input array. Extra NPZ keys are rejected. Loading never enables NumPy pickle support.

## Coordinates, bins and classes

The fixed target radius is `R=72 µm`. Ring edges are `[0,0.5,1,2,3] × R`; the last context radius is 216 µm. The inner two rings define the target neighbourhood. A boundary distance belongs to the inner ring (`distance <= outer_edge`), with distinct co-located cells retained in ring zero. The query object itself is removed by its ID.

For outward normal angle θ, canonical coordinates are `(x cos θ + y sin θ, -x sin θ + y cos θ)`. Sector zero starts at the canonical +x ray, and the sixteen sectors progress counterclockwise with half-open angular intervals of width 2π/16. Ray comparisons preserve the exact bin convention at axis/diagonal boundaries.

Class order is **immune, endothelial, stroma, epithelial**. Unknown class −1 cells affect the visible hull and centroid but are absent from all four count channels. The output is conditional on the four known classes; it is not a fraction of all unidentified cells.

## The seven query features

For valid queries:

1. First-boundary inward distance divided by R, clipped to [0,3]. For a halfplane this is `(cut − canonical_x)/R`. For a corner use its first boundary only; the complete two-boundary exposure is encoded separately in `crop_coverage`.
2. Canonical query x relative to the mean canonical coordinates of **all** visible cells, divided by R.
3. The corresponding centred canonical y coordinate divided by R.
4–7. The query-type one-hot vector. The pretrained epithelial model requires `[0,0,0,1]`.

The current operator's learned coefficients consume feature 1 and the polar arrays; features 2–7 remain required for a faithful, validated representation. They are not alternative free inputs for changing the target centre type.

## Exposure and visible support

The cell preparer uses analytic disk/annular-sector intersection areas. Full bins are represented by exactly 1; a geometrically partial bin that would round to 1 is instead represented by the next smaller float32 value. This preserves the exact-full-observation readout branch. Corner coverage is the intersection of the two known halfplanes, not a product of their separate area fractions.

The visible support hull uses all observed coordinates, including unknown types and the query centre. It is centred and scaled in the same frame as the query features. Do not substitute the complete-reference hull, a guessed tissue mask or a hull computed from hidden cells.

## Outputs

Prediction NPZ files contain `composition` `[W,Q,4]`, `missing_counts` `[W,Q,4]`, `missing_counts_by_bin` `[W,Q,2,16,4]`, `field_density` `[W,Q,4,16,4]`, and `valid` `[W,Q]`. Density is expected cell count per area unit πR²; bin area in those units is `(r_outer²-r_inner²)/16`. Missing count expectations integrate this field over the unobserved fraction of the inner two rings.

Only `valid=True` slots are predictions. Padded slots have uniform composition and zero field/expected missing counts. If the target is completely observed, composition equals normalized observed counts; a count-empty complete target receives the documented uniform fallback. These are conditional point estimates, not calibrated confidence intervals.
