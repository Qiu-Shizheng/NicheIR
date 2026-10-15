"""Deterministic synthetic cells for a complete offline tutorial."""
from __future__ import annotations
from pathlib import Path
import csv,json
import numpy as np
from .preprocessing import Cells,HalfPlane,Corner,prepare_cells
from .api import load_pretrained


def synthetic_cells(geometry: str='halfplane',seed: int=42) -> tuple[Cells,tuple[str,...]]:
    """Generate a synthetic visible field; no patient or experimental data."""
    if geometry not in {'halfplane','corner'}:raise ValueError('geometry must be halfplane or corner')
    rng=np.random.default_rng(seed);xy=rng.uniform(-320,320,(2400,2));labels=rng.choice([-1,0,1,2,3],size=len(xy),p=[.04,.29,.07,.22,.38])
    anchors=np.array([[-14,-13],[-22,-37],[-56,-28],[-31,-60],[-8,-55],[-62,-7],[-47,-41],[-18,-44]],dtype=float)
    visible=(xy[:,0]<=0)&((xy[:,1]<=0) if geometry=='corner' else True)
    xy=np.concatenate([xy[visible],anchors]);labels=np.concatenate([labels[visible],np.full(len(anchors),3)])
    ids=np.array([f'cell_{i:05d}' for i in range(len(xy))]);return Cells(ids,xy,labels),tuple(ids[-len(anchors):])


def run_example(output_dir: str | Path,geometry: str='halfplane') -> dict:
    """Write synthetic CSV, query IDs, prepared NPZ and pretrained predictions."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    cells,queries=synthetic_cells(geometry)
    with (output/'cells.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cell_id','x','y','class_id']);w.writerows((i,x,y,int(k)) for i,(x,y),k in zip(cells.cell_id,cells.xy,cells.class_id))
    with (output/'queries.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cell_id']);w.writerows((q,) for q in queries)
    boundary=HalfPlane(0) if geometry=='halfplane' else Corner(0,0)
    batch=prepare_cells(cells,queries,boundary);batch.save(output/'inputs.npz');batch.save_query_index(output/'query_index.csv')
    result=load_pretrained().predict(batch.inputs);result.save(output/'predictions.npz')
    report=dict(geometry=geometry,seed=42,synthetic=True,visible_cells=len(cells.cell_id),queries=len(queries),classes=list(load_pretrained().class_names),max_composition_sum_error=float(abs(result.composition[result.valid].sum(-1)-1).max()))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


def synthetic_training_batch(geometry: str='halfplane',seed: int=42):
    """Return visible-only model inputs and separate synthetic hidden-bin labels.

    The generated complete cloud supplies supervision only. No hidden coordinate
    or label is passed to the feature builder or model.
    """
    from ._geometry import polar_count
    cells,queries=synthetic_cells(geometry,seed)
    boundary=HalfPlane(0) if geometry=='halfplane' else Corner(0,0)
    batch=prepare_cells(cells,queries,boundary)
    rng=np.random.default_rng(seed);xy=rng.uniform(-320,320,(2400,2));labels=rng.choice([-1,0,1,2,3],size=len(xy),p=[.04,.29,.07,.22,.38])
    visible=(xy[:,0]<=0)&((xy[:,1]<=0) if geometry=='corner' else True)
    hidden_xy=xy[~visible];hidden_types=labels[~visible]
    targets=np.zeros((*batch.inputs['valid'].shape,2,16,4),dtype=np.float32)
    for slot,query_id in enumerate(queries):
        centre=cells.xy[np.flatnonzero(cells.cell_id==query_id)[0]]
        targets[0,slot]=polar_count((hidden_xy-centre)/72.,hidden_types,0.)[:2]
    return batch,targets


def run_training_example(output_dir: str | Path,epochs: int=2) -> dict:
    """Run a short supervised fit on synthetic cells and reload its checkpoint."""
    from .training import TrainingConfig,fit
    from .api import load_checkpoint
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    batch,targets=synthetic_training_batch();batch.save(output/'inputs.npz')
    np.savez_compressed(output/'targets.npz',hidden_counts=targets)
    result=fit(batch.inputs,targets,config=TrainingConfig(epochs=epochs))
    result.save(output/'trained.pt')
    expected=result.predictor.predict(batch.inputs).composition
    restored=load_checkpoint(output/'trained.pt').predict(batch.inputs)
    if not np.array_equal(expected,restored.composition):raise RuntimeError('Checkpoint round-trip changed predictions')
    restored.save(output/'predictions.npz')
    report=dict(synthetic=True,seed=42,epochs=epochs,queries=int(batch.inputs['valid'].sum()),checkpoint_round_trip=True,final_loss=result.history[-1]['loss'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
