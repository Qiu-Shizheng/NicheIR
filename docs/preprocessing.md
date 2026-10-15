# From annotated cells to model inputs

## Define the observed field

Supply one coordinate system, in micrometres, containing only cells genuinely observed under the stated acquisition mask. Keep unknown cells as class −1 so that they contribute to the visible support hull. At least three non-collinear observed points are required. Cells are ordered lexicographically by ID before deterministic feature construction; the requested query order is preserved. IDs must be unique; IDs are used to exclude the centre object, not to infer patient identities or cell types.

The declared halfplane or corner must describe acquisition visibility throughout each query’s 216 µm context. Additional unmodelled field edges, holes or disjoint tiles are not represented by this preparer.

No coordinate rescaling, registration, phenotype classification, segmentation or boundary estimation is performed. Map your annotation vocabulary to the four supported lineages before using the package. A missing marker panel cannot be repaired by labelling an unmeasured class as another lineage.

## Halfplane geometry

`HalfPlane(cut, angle)` keeps canonical x ≤ cut, where canonical x is `x*cos(angle)+y*sin(angle)`. The cut and input coordinates use the same physical origin. An angle of 0 means x ≤ cut; π/2 means y ≤ cut. The query must be on the visible side. A query at least 72 µm inward has complete target visibility, although its wider context may still be truncated.

```python
import math
from nicheir import HalfPlane, prepare_cells, read_cells_csv

cells = read_cells_csv("visible_cells.csv")
batch = prepare_cells(cells, ["epithelial_1"], HalfPlane(cut=100.0, angle=math.pi/2))
batch.save("inputs.npz")
batch.save_query_index("query_index.csv")
```

## Orthogonal corner geometry

`Corner(cut_x, cut_y, orientation)` keeps both canonical x ≤ cut_x and canonical y ≤ cut_y. The supported orientation is an integer quarter turn:

| Orientation | Canonical coordinates | Visible region with both cuts zero |
|---|---|---|
| 0 | `(x,y)` | x ≤ 0, y ≤ 0 |
| 1 | `(y,−x)` | y ≤ 0, x ≥ 0 |
| 2 | `(−x,−y)` | x ≥ 0, y ≥ 0 |
| 3 | `(−y,x)` | y ≥ 0, x ≤ 0 |

Both query-to-boundary distances must be in [0,72) µm. Other corner rotations or queries farther from either edge are outside this preparer's supported corner contract. The exact intersection area is computed using the bounded polygon [−4,0]² in radius units, whose auxiliary far edges lie outside every supported query's 3R context. The first normal defines the canonical angular origin.

## Boundary ties

Closed acquisition boundaries can contain cells exactly on an angular ray of a polar bin with zero two-dimensional exposure. A point count in that zero-area bin is incompatible with the count-exposure model. The preparer raises a clear error for that query; it does not add an arbitrary epsilon or silently remove the query. Review coordinate quantization and acquisition geometry, or explicitly omit the unsupported query. Do not adjust coordinates merely to obtain a desired prediction.

## Query grouping and field extent

Queries are visible epithelial cells. Their order is preserved and more than 32 queries are divided into consecutive 32-slot windows. Each query is processed independently by NicheIR. All chunks use the same complete visible-field coordinates, centroid and hull, so changing chunk size does not change the feature definition. Do not prepare different query chunks from different subsets of visible cells.

Different fields must be prepared separately. The API deliberately does not group cells by inferred patient/field columns or stitch fields together. A supplied visible field's convex hull is a support proxy; it does not certify that a biological tissue boundary or a complete reference neighbourhood was observed.
