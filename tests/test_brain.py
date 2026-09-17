import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from flybrain import integrate,FlyBrain,Adapter

def small_run(weight=100.,sign=1.):
    # A -> B. One initial spike in A; B must wait for synaptic delay.
    v=np.array([-40.,-52.],np.float32);g=np.zeros(2,np.float32);ref=np.zeros(2,np.int32)
    ring=np.zeros((19,2),np.int32);nr=np.zeros(19,np.int32)
    p=np.array([0,1,1],np.int64);idx=np.array([1],np.int32);w=np.array([weight],np.float32)
    signs=np.array([sign,1.],np.float32);inputs=np.empty(0,np.int32)
    cursor,c1=integrate(v,g,ref,ring,nr,0,p,idx,w,signs,inputs,np.zeros((18,0),bool),.1,20.,5.,18,22)
    assert c1.tolist()==[1,0] and g[1]==0
    cursor,c2=integrate(v,g,ref,ring,nr,cursor,p,idx,w,signs,inputs,np.zeros((100,0),bool),.1,20.,5.,18,22)
    return v,g,c2

def test_delay_and_direction():
    v,g,c=small_run();assert c[1]>0 and c[0]==0

def test_inhibition():
    v,g,c=small_run(sign=-1.);assert c.sum()==0 and v[1]<-52

def test_disconnected_edge():
    v,g,c=small_run(weight=0.);assert c.sum()==0 and v[1]==-52

def test_exact_passive_decay():
    v=np.array([-50.],np.float32);g=np.array([3.],np.float32)
    integrate(v,g,np.zeros(1,np.int32),np.zeros((19,1),np.int32),np.zeros(19,np.int32),0,np.array([0,0],np.int64),np.empty(0,np.int32),np.empty(0,np.float32),np.ones(1,np.float32),np.empty(0,np.int32),np.zeros((10,0),bool),.1,20.,5.,18,22)
    expected=-52+2*math.exp(-1/20)+3*5/(5-20)*(math.exp(-1/5)-math.exp(-1/20))
    assert abs(v[0]-expected)<2e-5

def test_silent_readout_stops():
    a=Adapter();cmd,s=a.decode(np.zeros(166700,np.int32),{'operator':{'position':[0,0,0],'aim':[1,0,0]}},50)
    assert cmd['action']=='stop' and max(s.values())==0
