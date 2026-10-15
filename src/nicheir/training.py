"""Supervised NicheIR fitting on prepared inputs and separate hidden-count labels."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
from collections.abc import Mapping
import json,math
import numpy as np
from numpy.typing import NDArray
import torch
from .api import Predictor,load_pretrained,initialize
from .inputs import validate_inputs

@dataclass(frozen=True)
class TrainingConfig:
    """Fixed-length training controls; one epoch visits every supplied window."""
    epochs: int=20
    batch_windows: int=4
    learning_rate: float=3e-4
    weight_decay: float=1e-4
    gradient_clip: float=5.
    seed: int=42
    window_tv_weight: float=.5
    margin_weight: float=.1
    count_nll_weight: float=.05

    def validate(self) -> None:
        for key in ('epochs','batch_windows'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f'{key} must be a positive integer')
        if type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('seed must be an integer in [0,2**32)')
        for key in ('learning_rate','gradient_clip','weight_decay','window_tv_weight','margin_weight','count_nll_weight'):
            value=getattr(self,key)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError(f'Invalid {key}')
        if self.learning_rate==0 or self.gradient_clip==0:raise ValueError('learning_rate and gradient_clip must be positive')

@dataclass
class TrainingResult:
    """Final-epoch model, numeric epoch history and explicit fit configuration."""
    predictor: Predictor
    history: list[dict[str,float]]
    config: TrainingConfig
    initialization: str

    def save(self,path: str | Path) -> None:
        """Write a tensor-only checkpoint and a path-free JSON training summary."""
        path=Path(path)
        if path.suffix!='.pt':raise ValueError('Use a .pt checkpoint filename')
        self.predictor.save_checkpoint(path)
        path.with_suffix('.json').write_text(json.dumps(dict(config=asdict(self.config),initialization=self.initialization,
            checkpoint='final_epoch',history=self.history),indent=2)+'\n',encoding='utf-8')


def validate_targets(inputs: Mapping[str,NDArray],hidden_counts: NDArray) -> NDArray:
    """Validate hidden counts in the inner two rings, [W,Q,2,16,4].

    Targets are supervised labels, never additional model input features.
    A fully observed bin and any padded slot must have zero hidden counts.
    """
    x=validate_inputs(inputs);y=np.asarray(hidden_counts)
    expected=(*x['valid'].shape,2,16,4)
    if y.shape!=expected:raise ValueError(f'hidden_counts must have shape {expected}')
    if y.dtype.kind not in 'iuf' or not np.isfinite(y).all() or (y<0).any() or not np.equal(y,np.rint(y)).all() or (y>2**24).any():
        raise ValueError('hidden_counts must contain finite nonnegative integer counts <=2**24')
    if (y[~x['valid']]!=0).any():raise ValueError('Padded slots require zero targets')
    if ((x['crop_coverage'][...,:2,:]==1)&(y.sum(-1)>0)).any():raise ValueError('A fully observed bin cannot contain hidden target cells')
    total=x['local_observed_counts'].astype(np.float64)+y.sum((-3,-2),dtype=np.float64)
    if (total>2**24).any():raise ValueError('Total per-type counts exceed float32 exact integer range')
    return y.astype(np.float32,copy=True)


def poisson_nll(rate: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    """Exact Poisson NLL per query, including factorials, normalized by 128 bins/types."""
    if rate.shape!=target.shape or rate.shape[-3:]!=(2,16,4):raise ValueError('Expected matching [...,2,16,4] rate/target arrays')
    r=rate.double();y=target.double()
    if not torch.isfinite(r).all() or not torch.isfinite(y).all() or (r<0).any() or (y<0).any() or not torch.equal(y,y.round()):
        raise ValueError('Poisson rates must be finite nonnegative; targets must be integer counts')
    if ((r==0)&(y>0)).any():raise ValueError('Positive target at zero Poisson rate')
    log_r=torch.where(r>0,r,torch.ones_like(r)).log()
    return (r-y*log_r+torch.lgamma(y+1)).sum((-3,-2,-1))/128.


def _objective(output,x,y,config):
    counts=x['local_observed_counts']+y.sum((-3,-2));total=counts.sum(-1,keepdim=True)
    target=torch.where(total>0,counts/total.clamp_min(1),torch.full_like(counts,.25));p=output['local'];valid=x['valid']
    def average_query(value):return (value*valid).sum(-1)/valid.sum(-1)
    local=average_query(.5*(p-target).abs().sum(-1))
    pmean=(p*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
    tmean=(target*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
    aggregate=.5*(pmean-tmean).abs().sum(-1)
    top=target.topk(2,dim=-1);gap=top.values[...,0]-top.values[...,1]
    winning=p.gather(-1,top.indices[...,:1]).squeeze(-1)
    competitors=p.scatter(-1,top.indices[...,:1],-1).max(-1).values
    margin=average_query(torch.relu(gap-(winning-competitors))*(gap>0))
    count=average_query(poisson_nll(output['missing_bins'],y))
    objective=local+config.window_tv_weight*aggregate+config.margin_weight*margin+config.count_nll_weight*count
    return objective,dict(local_tv=local,window_tv=aggregate,margin=margin,count_nll=count)


def fit(inputs: Mapping[str,NDArray],hidden_counts: NDArray,*,config: TrainingConfig | None=None,
        initialization: str='pretrained',device: str='cpu',window_weights: NDArray | None=None) -> TrainingResult:
    """Train a fresh NicheIR instance for a fixed number of epochs.

    ``hidden_counts`` is [W,Q,2,16,4], derived from separate complete references.
    Defaults fine-tune bundled weights. ``initialization='random'`` starts anew.
    Each epoch shuffles windows using a local seed; there is no early stopping,
    validation-set selection, patient splitting or automatic data balancing.
    Optional positive window weights are normalized once over the training set.
    """
    config=TrainingConfig() if config is None else config;config.validate()
    x=validate_inputs(inputs);y=validate_targets(x,hidden_counts);n=len(x['valid'])
    if initialization not in {'pretrained','random'}:raise ValueError('initialization must be pretrained or random')
    weights=np.ones(n,dtype=np.float64) if window_weights is None else np.asarray(window_weights,dtype=np.float64)
    if weights.shape!=(n,) or not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('window_weights must be finite positive [W]')
    weights=weights/weights.mean()
    predictor=load_pretrained(device=device) if initialization=='pretrained' else initialize(seed=config.seed,device=device)
    model=predictor.model;model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    rng=np.random.default_rng(config.seed);history=[];updates=0
    for epoch in range(config.epochs):
        order=rng.permutation(n);totals=dict(loss=0.,local_tv=0.,window_tv=0.,margin=0.,count_nll=0.)
        for start in range(0,n,config.batch_windows):
            ix=order[start:start+config.batch_windows]
            xx={k:torch.from_numpy(v[ix]).to(predictor.device) for k,v in x.items()};yy=torch.from_numpy(y[ix]).to(predictor.device)
            ww=torch.as_tensor(weights[ix],dtype=torch.float64,device=predictor.device)
            optimizer.zero_grad(set_to_none=True);per_window,parts=_objective(model(xx),xx,yy,config);loss=(per_window*ww).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True);optimizer.step();updates+=1
            totals['loss']+=float((per_window.detach()*ww).sum())
            for key,value in parts.items():totals[key]+=float((value.detach()*ww).sum())
        history.append(dict(epoch=epoch+1,updates=updates,**{k:v/n for k,v in totals.items()}))
    model.eval();return TrainingResult(predictor,history,config,initialization)
