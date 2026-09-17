import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from learning_brain import LearningBrain

N=80;MOTOR=np.arange(60,72,dtype=np.int32);INPUTS=np.arange(0,16,dtype=np.int32)
DYNAMIC=('v','g','refractory','ring','ring_counts','gains','pretrace','posttrace','prelast','postlast','silence')

def graph():
    rng=np.random.default_rng(11);indptr=[0];indices=[]
    for i in range(N):
        targets=np.sort(rng.choice(N,size=int(rng.integers(6,18)),replace=False))
        targets=targets[targets!=i];indices.extend(targets.tolist());indptr.append(len(indices))
    counts=rng.integers(5,40,size=len(indices)).astype(np.int32)
    signs=np.where(rng.random(N)<.3,-1.,1.).astype(np.float32)
    return (np.arange(1000,1000+N,dtype=np.int64),np.array(indptr,np.int64),np.array(indices,np.int32),counts,signs)

def make(restricted):
    b=LearningBrain(graph=graph(),seed=5)
    if restricted:b.set_plastic_posts(MOTOR)
    b.plasticity=True;return b

def drive(b,ticks,seed=3,start=0):
    rng=np.random.default_rng(seed);records=[]
    credit=np.linspace(-1,1,len(MOTOR)).astype(np.float32)
    for tick in range(start,start+ticks):
        pulses=rng.random((50,len(INPUTS)))<.25
        counts,_=b.step(INPUTS,np.zeros(len(INPUTS)),5.,pulses=pulses)
        snapshot=b.capture_motor_eligibility(MOTOR)
        update=b.reward_motor(snapshot,.6*(-1)**tick,MOTOR,credit)
        records.append((counts,snapshot,update))
    return records

def assert_same_motor_state(full,restricted):
    for name in DYNAMIC:assert np.array_equal(getattr(full,name),getattr(restricted,name)),name
    assert full.cursor==restricted.cursor
    motor_edges=restricted.plastic_post[np.asarray(full.indices,dtype=np.int64)]
    for name in ('eligibility','eligibility_last','listed'):
        assert np.array_equal(getattr(full,name)[motor_edges],getattr(restricted,name)[motor_edges]),name
        assert not getattr(restricted,name)[~motor_edges].any(),name

def assert_same_records(expected,actual):
    assert len(expected)==len(actual)
    for (c1,s1,u1),(c2,s2,u2) in zip(expected,actual):
        assert np.array_equal(c1,c2)
        for key in ('edges','posts','values'):assert np.array_equal(s1[key],s2[key]),key
        assert s1['cursor']==s2['cursor'] and u1==u2

def test_restricted_eligibility_is_bit_identical_for_motor_targets():
    full=make(False);restricted=make(True)
    expected=drive(full,8);actual=drive(restricted,8)
    assert sum(r[0].sum() for r in expected)>0,'synthetic graph must spike'
    assert sum(len(r[1]['edges']) for r in expected)>0,'motor credit must be exercised'
    assert_same_records(expected,actual);assert_same_motor_state(full,restricted)
    assert restricted.active_count<full.active_count

def test_whole_graph_checkpoint_loads_into_restricted_brain(tmp_path):
    full=make(False);drive(full,4);full.save(tmp_path/'whole.npz')
    restricted=make(True);restricted.load(tmp_path/'whole.npz')
    assert np.all(restricted.plastic_post[np.asarray(restricted.indices,dtype=np.int64)[restricted.active[:restricted.active_count]]])
    expected=drive(full,4,seed=9,start=4);actual=drive(restricted,4,seed=9,start=4)
    assert_same_records(expected,actual);assert_same_motor_state(full,restricted)

def test_restricted_brain_rejects_global_rule_and_invalid_targets():
    b=make(True);drive(b,2)
    with pytest.raises(ValueError,match='Global R-STDP'):b.reward(.5)
    with pytest.raises(ValueError,match='plastic target'):b.set_plastic_posts([N+1])

def test_reset_keeps_restriction_and_clears_eligibility():
    b=make(True);drive(b,3);b.reset(4)
    assert b.restricted_plasticity and b.plastic_post[MOTOR].all() and b.active_count==0
