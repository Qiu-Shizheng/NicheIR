"""Validate the visible-input archive contract before inference."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

Inputs: TypeAlias = dict[str, NDArray]
INPUT_KEYS = frozenset({'polar_counts','crop_coverage','visible_hull_support','local_observed_counts','query_features','valid'})
CLASS_NAMES = ('immune','endothelial','stroma','epithelial')


def validate_inputs(values: Mapping[str, NDArray]) -> Inputs:
    """Return independent float32/bool arrays or raise a descriptive ValueError.

    Counts are nonnegative integers encoded as float32. Invalid slots contain
    zero data. Each window has 1–32 slots and at least one valid query.
    """
    if set(values)!=INPUT_KEYS:
        raise ValueError(f'Expected six input arrays; missing={sorted(INPUT_KEYS-set(values))}, extra={sorted(set(values)-INPUT_KEYS)}')
    a={k:np.asarray(v) for k,v in values.items()}
    c=a['polar_counts']
    if c.ndim!=5 or c.shape[-3:]!=(4,16,4):raise ValueError('polar_counts must have shape [windows, slots, 4, 16, 4]')
    w,q=c.shape[:2]
    if w<1 or not 1<=q<=32:raise ValueError('Require at least one window and 1–32 slots per window')
    expected={'crop_coverage':(w,q,4,16),'visible_hull_support':(w,q,4,16),'local_observed_counts':(w,q,4),'query_features':(w,q,7),'valid':(w,q)}
    for k,shape in expected.items():
        if a[k].shape!=shape:raise ValueError(f'{k} must have shape {shape}')
    for k,v in a.items():
        if v.dtype.kind not in 'buif' or not np.isfinite(v).all():raise ValueError(f'{k} must contain finite numeric values')
    if not np.isin(a['valid'],[0,1]).all():raise ValueError('valid must be boolean or contain only 0/1')
    valid=a['valid'].astype(bool)
    if not valid.any(axis=1).all():raise ValueError('Every window must contain at least one valid query')
    for k in ['polar_counts','local_observed_counts']:
        if (a[k]<0).any() or not np.equal(a[k],np.floor(a[k])).all():raise ValueError(f'{k} must contain nonnegative integer counts')
        if (a[k]>2**24).any():raise ValueError(f'{k} exceeds the exact float32 integer range')
    for k in ['crop_coverage','visible_hull_support']:
        if ((a[k]<0)|(a[k]>1)).any():raise ValueError(f'{k} must be in [0,1]')
    if (a['visible_hull_support']>a['crop_coverage']+2e-6).any():raise ValueError('Visible hull support must be a subset of acquisition coverage')
    if not np.array_equal(c[:,:,:2].sum(axis=(2,3)),a['local_observed_counts']):raise ValueError('local_observed_counts must equal the inner two polar rings')
    if ((a['crop_coverage']==0)&(c.sum(axis=-1)>0)).any():raise ValueError('A zero-exposure bin contains observed cells; this boundary query is unsupported')
    f=a['query_features']
    if ((f[...,0][valid]<0)|(f[...,0][valid]>3)).any():raise ValueError('Normalized first-boundary distance must be in [0,3]')
    if not np.array_equal(f[...,3:][valid],np.tile([0,0,0,1],(int(valid.sum()),1))):raise ValueError('The pretrained model supports epithelial query centres only')
    for k in INPUT_KEYS-{'valid'}:
        if np.any(a[k][~valid]!=0):raise ValueError(f'Invalid slots must have zero {k}')
    out={k:np.array(v,dtype=bool if k=='valid' else np.float32,copy=True) for k,v in a.items()}
    # Re-check after conversion: positive exposure must never silently round to zero.
    if ((a['crop_coverage']<1)&(out['crop_coverage']==1)).any():raise ValueError('Partial exposure rounds to full in float32; use the geometric preparer to preserve full-bin semantics')
    if ((out['crop_coverage']==0)&(out['polar_counts'].sum(-1)>0)).any():raise ValueError('Exposure underflows to zero in float32')
    if not np.array_equal(out['polar_counts'][:,:,:2].sum((2,3)),out['local_observed_counts']):raise ValueError('Count sum is not exactly representable in float32')
    return out


def load_inputs(path: str | Path) -> Inputs:
    """Load an NPZ without pickle, then validate its six arrays."""
    with np.load(path,allow_pickle=False) as z:
        return validate_inputs({k:z[k] for k in z.files})


def save_inputs(path: str | Path, values: Mapping[str, NDArray]) -> None:
    """Write exactly the six validated visible arrays, with no identifiers."""
    np.savez_compressed(path,**validate_inputs(values))
