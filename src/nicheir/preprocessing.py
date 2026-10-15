"""Build the trained feature representation from one visible annotated field."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
import csv
import numpy as np
from numpy.typing import NDArray
from . import _geometry as geom
from .inputs import Inputs,validate_inputs,save_inputs

RADIUS_UM = 72.0

@dataclass(frozen=True)
class HalfPlane:
    """Visible side: x*cos(angle)+y*sin(angle) <= cut; radians and micrometres."""
    cut: float
    angle: float = 0.0

@dataclass(frozen=True)
class Corner:
    """Two upper halfplanes in a canonical frame rotated orientation*90 degrees.

    orientation is 0,1,2,3. Both query-to-boundary distances must be <72 um.
    """
    cut_x: float
    cut_y: float
    orientation: int = 0

@dataclass(frozen=True)
class Cells:
    """Observed cells from exactly one field, including unknown class -1."""
    cell_id: NDArray
    xy: NDArray
    class_id: NDArray

@dataclass
class PreparedBatch:
    """Six arrays plus the explicit user-supplied query-to-slot mapping."""
    inputs: Inputs
    query_ids: tuple[str,...]

    def save(self,path: str | Path) -> None:
        save_inputs(path,self.inputs)

    def save_query_index(self,path: str | Path) -> None:
        with Path(path).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['window','slot','cell_id'])
            w.writerows((i//32,i%32,identifier) for i,identifier in enumerate(self.query_ids))


def read_cells_csv(path: str | Path) -> Cells:
    """Read cell_id,x,y,class_id. Coordinates must already be micrometres.

    Additional columns are ignored. No image segmentation or type inference
    is performed, and different fields must not be concatenated.
    """
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'cell_id','x','y','class_id'}<=set(reader.fieldnames or []):
            raise ValueError('CSV requires cell_id,x,y,class_id columns')
        rows=list(reader)
    try:
        ids=np.array([r['cell_id'] for r in rows],dtype=str)
        xy=np.array([[float(r['x']),float(r['y'])] for r in rows],dtype=np.float64)
        labels=np.array([float(r['class_id']) for r in rows])
    except (TypeError,ValueError) as e:raise ValueError('CSV coordinates and class_id must be numeric') from e
    return Cells(ids,xy,labels)


def _corner_coverage(distances: NDArray) -> NDArray:
    d=np.asarray(distances,dtype=np.float64)
    if d.ndim!=2 or d.shape[1]!=2 or not np.isfinite(d).all() or not ((d>=0)&(d<1)).all():
        raise ValueError('Corner query distances must both be in [0,72) micrometres')
    polygon=np.asarray([[-4.,-4.],[0.,-4.],[0.,0.],[-4.,0.]])
    raw=geom.analytic_hull_support(polygon,-d)
    first=geom.encoded_crop_coverage(d[:,0]);second=np.roll(geom.encoded_crop_coverage(d[:,1]),4,axis=-1)
    full=(first==1)&(second==1)
    encoded=raw.astype(np.float32);encoded[full]=1.
    encoded[(~full)&(encoded==1)]=np.nextafter(np.float32(1),np.float32(0))
    return encoded


def prepare_cells(cells: Cells,query_ids: Sequence[str],acquisition: HalfPlane | Corner) -> PreparedBatch:
    """Prepare visible epithelial queries using the fixed 72-um representation.

    All supplied cells must be visible under acquisition. Unknown (-1) cells
    affect the observed support hull and centroid, but do not enter type counts.
    The centre object is excluded by ID; distinct co-located cells are retained.
    Queries are chunked into 32 slots without changing their supplied order.
    A zero-exposure/positive-count boundary conflict raises ValueError rather
    than changing geometry or silently deleting a requested query.
    """
    ids=np.asarray(cells.cell_id).astype(str);xy=np.asarray(cells.xy,dtype=np.float64);labels=np.asarray(cells.class_id)
    if ids.ndim!=1 or xy.shape!=(len(ids),2) or labels.shape!=(len(ids),):raise ValueError('Cells require matching IDs, [N,2] coordinates and [N] classes')
    if len(ids)<3 or len(set(ids))!=len(ids) or any(not s for s in ids):raise ValueError('At least three cells with nonempty unique IDs are required')
    if not np.isfinite(xy).all() or labels.dtype.kind not in 'buif' or not np.isfinite(labels).all() or not np.isin(labels,[-1,0,1,2,3]).all():raise ValueError('Finite coordinates and integer classes -1,0,1,2,3 are required')
    order=np.argsort(ids,kind='stable');ids=ids[order];xy=xy[order];labels=labels[order].astype(np.int64)
    queries=tuple(str(q) for q in query_ids)
    if not queries or len(set(queries))!=len(queries):raise ValueError('Provide at least one unique query ID')
    lookup={s:i for i,s in enumerate(ids)}
    if any(q not in lookup for q in queries):raise ValueError('Every query ID must identify a visible cell')
    qi=np.array([lookup[q] for q in queries]);
    if not (labels[qi]==3).all():raise ValueError('Every query must be an epithelial cell (class_id=3)')
    if isinstance(acquisition,HalfPlane):
        theta=float(acquisition.angle);cut=float(acquisition.cut)
        if not np.isfinite([theta,cut]).all():raise ValueError('Boundary angle and cut must be finite')
        z=geom.canonical_coordinates(xy,theta)
        if (z[:,0]>cut).any():raise ValueError('Input includes cells outside the visible halfplane')
    elif isinstance(acquisition,Corner):
        if type(acquisition.orientation) is not int or acquisition.orientation not in range(4):raise ValueError('Corner orientation must be 0,1,2 or 3')
        theta=acquisition.orientation*np.pi/2;cut=float(acquisition.cut_x)
        cuts=np.array([cut,float(acquisition.cut_y)])
        if not np.isfinite(cuts).all():raise ValueError('Corner cuts must be finite')
        z=geom.canonical_coordinates(xy,theta)
        if (z>cuts).any():raise ValueError('Input includes cells outside the visible corner intersection')
        distances=(cuts-z[qi])/RADIUS_UM
        if not ((distances>=0)&(distances<1)).all():raise ValueError('Corner query distances must both be in [0,72) micrometres')
    else:raise TypeError('acquisition must be HalfPlane or Corner')
    pieces=[]
    for start in range(0,len(queries),32):
        batch=queries[start:start+32]
        try:x,_=geom.observable_features(xy,labels,ids,batch,theta,cut,RADIUS_UM)
        except AssertionError as e:raise ValueError('Visible geometry violates the supported feature contract') from e
        if isinstance(acquisition,Corner):x['crop_coverage'][:len(batch)]=_corner_coverage(distances[start:start+32])
        x['valid']=x['valid'].astype(bool);pieces.append(x)
    arrays={k:np.stack([x[k] for x in pieces]) for k in pieces[0]}
    return PreparedBatch(validate_inputs(arrays),queries)
