import math
import numpy as np
import pytest
import torch
from nicheir import fit,TrainingConfig,load_checkpoint,load_pretrained,validate_targets
from nicheir.example import synthetic_training_batch
from nicheir.training import poisson_nll

@pytest.fixture(scope='module')
def batch():
    torch.set_num_threads(2)
    return synthetic_training_batch()

def test_exact_poisson():
    rate=torch.zeros(1,1,2,16,4,dtype=torch.float64,requires_grad=True)
    target=torch.zeros_like(rate)
    assert poisson_nll(rate,target).item()==0
    target[0,0,0,0,0]=1
    with pytest.raises(ValueError,match='zero Poisson rate'):poisson_nll(rate,target)
    r=torch.ones_like(target);r[0,0,0,0,0]=2
    target[0,0,0,0,0]=3
    expected=(127+2-3*math.log(2)+math.lgamma(4))/128
    assert abs(poisson_nll(r,target).item()-expected)<1e-12
    rate2=torch.full_like(target,1e-40,requires_grad=True)
    poisson_nll(rate2,target).sum().backward()
    assert torch.isfinite(rate2.grad).all()

def test_training_determinism_and_checkpoint(batch,tmp_path):
    data,y=batch
    original=load_pretrained();before={k:v.clone() for k,v in original.model.state_dict().items()}
    config=TrainingConfig(epochs=2)
    first=fit(data.inputs,y,config=config);second=fit(data.inputs,y,config=config)
    assert first.history==second.history
    assert first.history[-1]['epoch']==2 and first.history[-1]['updates']==2
    assert all(torch.equal(v,second.predictor.model.state_dict()[k]) for k,v in first.predictor.model.state_dict().items())
    assert any(not torch.equal(v,before[k]) for k,v in first.predictor.model.state_dict().items())
    assert all(torch.equal(v,load_pretrained().model.state_dict()[k]) for k,v in before.items())
    first.save(tmp_path/'adapted.pt');restored=load_checkpoint(tmp_path/'adapted.pt')
    assert np.array_equal(first.predictor.predict(data.inputs).composition,restored.predict(data.inputs).composition)
    state=torch.load(tmp_path/'adapted.pt',weights_only=True)
    assert all(isinstance(v,torch.Tensor) for v in state.values())
    state['bin_area'][0,0]*=2;torch.save(state,tmp_path/'changed.pt')
    with pytest.raises(ValueError,match='fixed geometry'):load_checkpoint(tmp_path/'changed.pt')

def test_target_validation(batch):
    data,y=batch
    assert np.array_equal(validate_targets(data.inputs,y),y)
    bad=y.copy();bad[0,-1,0,0,0]=1
    with pytest.raises(ValueError,match='Padded'):validate_targets(data.inputs,bad)
    bad=y.copy();full=np.argwhere(data.inputs['crop_coverage'][0,0,:2]==1)[0];bad[0,0,int(full[0]),int(full[1]),0]=1
    with pytest.raises(ValueError,match='fully observed'):validate_targets(data.inputs,bad)
    bad=y.copy();bad[0,0,0,0,0]=.5
    with pytest.raises(ValueError,match='integer'):validate_targets(data.inputs,bad)
    with pytest.raises(ValueError,match='positive'):fit(data.inputs,y,config=TrainingConfig(epochs=0))
    with pytest.raises(ValueError,match='window_weights'):fit(data.inputs,y,window_weights=[0])

def test_random_initialization_fit(batch):
    data,y=batch
    result=fit(data.inputs,y,config=TrainingConfig(epochs=1),initialization='random')
    assert result.initialization=='random' and math.isfinite(result.history[0]['loss'])
    result.predictor.predict(data.inputs)
