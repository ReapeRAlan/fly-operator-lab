from pathlib import Path
import sys,json,time,hashlib
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_brain import LearningBrain
from learning_adapter import SensoryEncoder,NeuralReadout

def fingerprint(b):
    h=hashlib.sha256()
    for name in b.array_state:h.update(memoryview(np.ascontiguousarray(getattr(b,name))).cast('B'))
    h.update(b.active[:b.active_count].tobytes());h.update(str(b.cursor).encode());h.update(json.dumps(b.rng.bit_generator.state).encode())
    return h.hexdigest()

def main():
    start=time.time();b=LearningBrain();e=SensoryEncoder();readout=NeuralReadout()
    assert b.n==166700 and b.edges==25582938 and len(readout.indices)==1314
    rates=np.full(len(e.ids),5.);rates[:32]=100.
    frozen=b.step(e.ids,rates,50.)[1]
    b.plasticity=True;plastic=b.step(e.ids,rates,50.)[1];change=b.reward(.5)
    assert change['changed_edges']>0
    checkpoint=ROOT/'work/learning/full_validation.npz';b.save(checkpoint,{'test':'full_continuation'})
    expected=b.step(e.ids,rates,50.)[0];b.reward(-.2);expected_hash=fingerprint(b)
    b.load(checkpoint);actual=b.step(e.ids,rates,50.)[0];b.reward(-.2)
    assert np.array_equal(expected,actual) and fingerprint(b)==expected_hash
    result=dict(passed=True,neurons=b.n,edges=b.edges,readout_neurons=len(readout.indices),graph_hash=b.graph_hash,
      frozen=frozen,plastic=plastic,update=change,checkpoint_exact_continuation=True,
      rss_mib=psutil.Process().memory_info().rss/2**20,wall_seconds=time.time()-start)
    (ROOT/'outputs/learning_full_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
