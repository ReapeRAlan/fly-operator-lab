import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from learning_brain import LearningBrain,Dynamics,modulate

def brain():
    graph=(np.array([10,20,30],np.int64),np.array([0,1,2,2],np.int64),np.array([1,2],np.int32),np.array([1,1],np.int32),np.ones(3,np.int8))
    return LearningBrain(graph=graph,seed=7)

def stimulate(b,schedule,length=100):
    pulses=np.zeros((length,3),bool)
    for tick,neuron in schedule:pulses[tick,neuron]=True
    return b.step([0,1,2],[0,0,0],length*.1,pulses=pulses,record=True)

def test_directed_stdp_arrival_then_post_potentiates():
    b=brain();b.plasticity=True
    stimulate(b,[(0,0),(30,1)])
    assert b.eligibility[0]>0
    assert b.reward(1)['changed_edges']>0 and b.gains[0]>1
    assert len(b.indices)==2 and np.array_equal(b.counts,[1,1])

def test_post_then_arrival_depresses():
    b=brain();b.plasticity=True
    stimulate(b,[(0,1),(30,0)])
    assert b.eligibility[0]<0
    b.reward(1);assert b.gains[0]<1

def test_reward_sign_and_frozen():
    b=brain();b.plasticity=True;stimulate(b,[(0,0),(30,1)])
    b.reward(-1);assert b.gains[0]<1
    b.plasticity=False;old=b.gains.copy();b.reward(1);assert np.array_equal(old,b.gains)

def test_checkpoint_exact_continuation(tmp_path):
    b=brain();b.plasticity=True
    b.step([0,1],[80,90],3.);b.reward(.3)
    b.save(tmp_path/'state.npz',{'filters':[1,2,3]})
    expected=b.step([0,1],[80,90],50.,record=True)[0];b.reward(.4)
    c=brain();assert c.load(tmp_path/'state.npz')['filters']==[1,2,3]
    actual=c.step([0,1],[80,90],50.,record=True)[0];c.reward(.4)
    assert np.array_equal(expected,actual)
    for name in b.array_state:assert np.array_equal(getattr(b,name),getattr(c,name)),name
    assert b.cursor==c.cursor and np.array_equal(b.active[:b.active_count],c.active[:c.active_count])

def test_episode_reset_retains_learning_but_clears_activity():
    b=brain();b.plasticity=True;stimulate(b,[(0,0),(30,1)]);b.reward(1)
    old=b.gains.copy();b.reset(9)
    assert np.array_equal(old,b.gains) and b.cursor==0 and b.active_count==0
    assert not b.eligibility.any() and not b.ring_counts.any()

def test_gain_bounds_and_inhibition_preserved():
    b=brain();b.gains[:]=3.99;b.eligibility[:]=1e9;b.active[:2]=[0,1];b.active_count=2;b.plasticity=True
    b.reward(1);assert np.all(b.gains==4)
    b.reward(-1);assert np.all(b.gains==.25)

def test_intervention_silence_blocks_output():
    b=brain();b.silence[:]=True
    counts,_=stimulate(b,[(0,0),(30,1)])
    assert counts.sum()==0

def test_bad_graph_checkpoint_rejected(tmp_path):
    b=brain();b.save(tmp_path/'state.npz');b.graph_hash='changed'
    with pytest.raises(ValueError,match='graph'):b.load(tmp_path/'state.npz')
