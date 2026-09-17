from pathlib import Path
from dataclasses import replace
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from learning_brain import LearningBrain,Dynamics
from learning_adapter import SensoryEncoder
def main():
    b=LearningBrain();e=SensoryEncoder();rates=np.full(len(e.ids),5.);rates[:32]=100
    b.step(e.ids,rates,50);path=ROOT/'work/learning/calibration_base.npz';b.save(path);rows=[]
    for eta in (1e-5,1e-4,1e-3):
        b.dynamics=Dynamics();b.load(path);b.dynamics=replace(b.dynamics,learning_rate=eta);b.plasticity=True
        counts,activity=b.step(e.ids,rates,50);update=b.reward(.5)
        rows.append({'eta':eta,'activity':activity,'update':update,'silent_fraction':float(np.mean(counts==0)),
          'near_max_spike_fraction':float(np.mean(counts>=21)),'gain_min':float(b.gains.min()),'gain_max':float(b.gains.max()),
          'finite':bool(np.isfinite(b.v).all() and np.isfinite(b.g).all()),'mean_firing_hz':float(counts.mean()*20)})
    out={'scope':'Short calibration under identical state, noise and 50 ms stimulation; not long-term physiological validation','seed':7,'selected_initial_eta':1e-4,'rows':rows}
    (ROOT/'outputs/plasticity_calibration_v2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
