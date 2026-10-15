"""Offline inference, model initialization and tensor-only checkpoint I/O."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from importlib.resources import files
import hashlib,io,json
import numpy as np
from numpy.typing import NDArray
import torch
from ._model import NicheIR
from .inputs import validate_inputs,load_inputs,CLASS_NAMES

@dataclass
class Prediction:
    """Four-class conditional compositions and nonnegative expected hidden counts.

    Arrays have leading [window,slot] dimensions. Only valid=True rows are
    predictions. Invalid slots use a uniform composition and zero hidden counts.
    """
    composition: NDArray
    missing_counts: NDArray
    missing_counts_by_bin: NDArray
    field_density: NDArray
    valid: NDArray

    def save(self,path: str | Path) -> None:
        np.savez_compressed(path,composition=self.composition,missing_counts=self.missing_counts,
                            missing_counts_by_bin=self.missing_counts_by_bin,field_density=self.field_density,valid=self.valid)

class Predictor:
    """Pretrained NicheIR inference. Instances retain no input data after calls."""
    class_names=CLASS_NAMES

    def __init__(self,model: NicheIR,device: torch.device):
        self.model=model.eval();self.device=device

    def predict(self,inputs: Mapping[str,NDArray],*,batch_windows: int=4) -> Prediction:
        """Validate, predict and return CPU arrays; no training or random sampling."""
        if type(batch_windows) is not int or batch_windows<1:raise ValueError('batch_windows must be a positive integer')
        arrays=validate_inputs(inputs);n=len(arrays['valid']);parts=[]
        with torch.inference_mode():
            for start in range(0,n,batch_windows):
                x={k:torch.from_numpy(v[start:start+batch_windows]).to(self.device) for k,v in arrays.items()}
                out=self.model(x);parts.append({k:v.cpu().numpy() for k,v in out.items()})
        collect=lambda key:np.concatenate([p[key] for p in parts],axis=0)
        return Prediction(collect('local'),collect('missing_rate'),collect('missing_bins'),collect('field_density'),arrays['valid'])

    def save_checkpoint(self,path: str | Path) -> None:
        """Save CPU tensors only, without data, paths, optimizer or user metadata."""
        torch.save({k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},path)

    def predict_npz(self,path: str | Path,*,output: str | Path | None=None,batch_windows: int=4) -> Prediction:
        """Predict a supported six-array NPZ, optionally saving a result archive."""
        result=self.predict(load_inputs(path),batch_windows=batch_windows)
        if output is not None:result.save(output)
        return result


def load_pretrained(*,device: str='cpu') -> Predictor:
    """Load the bundled seed42 weights without a download or global RNG changes."""
    root=files('nicheir').joinpath('weights');payload=root.joinpath('nicheir.pt').read_bytes()
    metadata=json.loads(root.joinpath('model.json').read_text())
    if hashlib.sha256(payload).hexdigest()!=metadata['sha256']:raise RuntimeError('Bundled model checksum mismatch')
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):raise RuntimeError('Expected a tensor-only state dictionary')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);model=NicheIR()
    model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters())!=2917:raise RuntimeError('Unexpected model architecture')
    target=torch.device(device);model.to(target)
    return Predictor(model,target)


def initialize(*,seed: int=42,device: str='cpu') -> Predictor:
    """Create an untrained instance without changing the caller's CPU RNG state."""
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('seed must be an integer in [0,2**32)')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=NicheIR()
    target=torch.device(device);return Predictor(model.to(target),target)


def load_checkpoint(path: str | Path,*,device: str='cpu') -> Predictor:
    """Load a tensor-only NicheIR checkpoint; reject changed fixed geometry."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Expected a tensor-only state dictionary')
    predictor=initialize(device=device)
    for key,value in state.items():
        if not torch.isfinite(value).all():raise ValueError(f'Nonfinite checkpoint tensor: {key}')
    for key,buffer in predictor.model.named_buffers():
        if key not in state or not torch.equal(buffer.cpu(),state[key]):raise ValueError(f'Checkpoint changed fixed geometry: {key}')
    predictor.model.load_state_dict(state,strict=True)
    return predictor
