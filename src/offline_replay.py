"""Open-loop replay of recorded engine observations through the complete brain.

Recorded clean observations are re-encoded with a chosen sensory schema and
integrated in their original order, so encoder variants can be compared on
identical trajectories without the game. The trajectory does not depend on the
network (open loop): this measures signal available to a readout, not competence.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math,sqlite3
import numpy as np
from learning_adapter import SensoryEncoder,NeuralReadout,ActionCatalog,bearing
from learning_env import ExerciseTeacher
ROOT=Path(__file__).resolve().parents[1]
MOVE_TOLERANCE=math.pi/4+1e-9


def load_episodes(run,stage='move',database=ROOT/'work/learning/experiments.sqlite'):
    """Group recorded steps into unique episodes ordered by record id (read-only)."""
    db=sqlite3.connect(f'file:{Path(database).as_posix()}?mode=ro',uri=True,timeout=15)
    try:
        rows=db.execute("""SELECT id,payload FROM steps WHERE run=? AND json_extract(payload,'$.stage')=?
          AND coalesce(json_extract(payload,'$.evaluation_preview'),0)=0 ORDER BY id""",(run,stage)).fetchall()
    finally:db.close()
    groups={}
    for ident,payload in rows:
        item=json.loads(payload)
        key=(item.get('game_session'),item.get('episode'),item.get('condition'),item.get('seed'),item.get('mission'))
        groups.setdefault(key,[]).append({'id':ident,'observation':item['observation'],'mask':item.get('pre_action',{}).get('mask'),
          'decision_required':bool(item.get('decision_required')),'condition':item.get('condition'),'seed':item.get('seed'),
          'split':item.get('split'),'mission':item.get('mission'),'sequence':item.get('sequence')})
    episodes=[];seen=set()
    for key,steps in groups.items():
        steps.sort(key=lambda s:s['sequence'])
        if [s['sequence'] for s in steps]!=list(range(steps[0]['sequence'],steps[0]['sequence']+len(steps))):continue
        # Frozen/adapter/combined share bit-identical first episodes; keep one copy.
        trajectory=hashlib.sha256(json.dumps([key[4]]+[s['observation']['operator'].get('position') for s in steps]).encode()).hexdigest()
        if trajectory in seen:continue
        seen.add(trajectory);episodes.append({'key':[str(k) for k in key],'trajectory_sha256':trajectory,'steps':steps})
    return episodes


def teacher_label(catalog,before,mask,step_index):
    """ExerciseTeacher on the pre-decision observation and the recorded action mask."""
    env=SimpleNamespace(obs=before,mission={'stage':before['stage'],'goal':before['goal'],'seed':0},steps=step_index,
                        action_masks=lambda:np.asarray(mask,bool),cancel_practiced=False,rescue_followed=False,shot_ever=False,reload_oriented=True)
    return int(ExerciseTeacher(catalog).choose(env))


def replay_episode(brain,encoder,readout,catalog,episode,seed):
    """Return decision rows (features, channel values, label, mask) and activity statistics."""
    steps=episode['steps'];brain.reset(seed);brain.plasticity=False;readout.reset()
    ceiling=18 # >=18 spikes in 50 ms is >=360 Hz, near the 2.3 ms refractory limit
    rows=[];ticks=0;dn_high=[];active=[];spikes=[]
    def integrate(observation):
        nonlocal ticks
        ids,rates,channels=encoder.encode(observation)
        counts,activity=brain.step(ids,rates,50.)
        features=readout.update(counts);dn=counts[readout.indices]
        dn_high.append(float(np.mean(dn>=ceiling)));active.append(activity['active_neurons']);spikes.append(activity['spikes']);ticks+=1
        return features,channels
    # The reset observation is not recorded; the first recorded outcome stands in for its 50 ms warm-up.
    features,channels=integrate(steps[0]['observation'])
    for index in range(1,len(steps)):
        before=steps[index-1]['observation'];step=steps[index]
        if step['decision_required'] and step['mask'] is not None:
            label=teacher_label(catalog,before,step['mask'],index)
            rows.append({'features':features,'channels':np.array([channels[k] for k in encoder.channels],np.float32),
                         'label':label,'mask':np.asarray(step['mask'],bool)})
        features,channels=integrate(step['observation'])
    return rows,{'ticks':ticks,'dn_near_ceiling_fraction':float(np.mean(dn_high)),'active_neurons_mean':float(np.mean(active)),
                 'spikes_mean':float(np.mean(spikes))}


def within_tolerance(catalog,predicted,label):
    a=catalog.entries[int(predicted)];b=catalog.entries[int(label)]
    if int(predicted)==int(label):return True
    if a['action']!='move' or b['action']!='move':return False
    return abs((a['angle']-b['angle']+math.pi)%(2*math.pi)-math.pi)<=MOVE_TOLERANCE


def _ridge_scores(train_x,train_y,test_x,classes,lam):
    mu=train_x.mean(0);sd=train_x.std(0)+1e-6;a=(train_x-mu)/sd;targets=np.eye(classes)[train_y]
    offset=targets.mean(0);alpha=np.linalg.solve(a@a.T+lam*np.eye(len(a)),targets-offset)
    return ((test_x-mu)/sd)@a.T@alpha+offset


def grouped_ridge_cv(x,y,masks,groups,catalog,lambdas=(1.,10.,100.,1000.,10000.),folds=5):
    """Episode-grouped CV; the ridge penalty is chosen by inner grouped CV on training folds only."""
    x=np.asarray(x,np.float64);y=np.asarray(y);masks=np.asarray(masks,bool);groups=np.asarray(groups)
    unique=np.unique(groups);order=np.random.default_rng(0).permutation(len(unique));assignment={g:i%folds for i,g in zip(order,unique)}
    fold=np.array([assignment[g] for g in groups]);classes=masks.shape[1];exact=0;tolerant=0;chosen=[]
    def predict(tr,te,lam):
        scores=_ridge_scores(x[tr],y[tr],x[te],classes,lam);scores[~masks[te]]=-np.inf;return scores.argmax(1)
    for k in range(folds):
        train=np.where(fold!=k)[0];test=np.where(fold==k)[0]
        if len(test)==0:continue
        inner=fold[train];best=None
        for lam in lambdas:
            hits=0
            for j in set(inner.tolist()):
                itr=train[inner!=j];ite=train[inner==j]
                hits+=sum(within_tolerance(catalog,p,t) for p,t in zip(predict(itr,ite,lam),y[ite]))
            if best is None or hits>best[0]:best=(hits,lam)
        chosen.append(best[1]);prediction=predict(train,test,best[1])
        exact+=int(np.sum(prediction==y[test]));tolerant+=sum(within_tolerance(catalog,p,t) for p,t in zip(prediction,y[test]))
    return {'exact':exact/len(y),'within_45deg':tolerant/len(y),'chosen_lambdas':chosen,'samples':int(len(y)),'episodes':int(len(unique))}
