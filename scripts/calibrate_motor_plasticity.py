"""Paired calibration of causal motor plasticity: direction (v1 vs v2) and learning-rate magnitude.

Protocol per (rule, learning rate), on identical replayed move episodes and noise seeds:
1. frozen replay with unit gains -> log pi(instructor action) at each decision (shared baseline);
2. plastic replay: each decision's eligibility snapshot is credited at the next decision with
   the gradient of log pi(instructor action) and a positive prediction error drawn from the
   magnitudes recorded in v3.1 internal learning;
3. frozen replay with the learned gains and the same seeds -> log pi(instructor action).
Open loop with always-positive credit: this checks that the rule moves the policy in the
credited direction and sizes the step. It is not a reinforcement-learning result.
"""
from pathlib import Path
import sys,json,time,argparse,sqlite3,math
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def recorded_prediction_errors(run='pilot-v3.1-curriculum'):
    db=sqlite3.connect(f'file:{(ROOT/"work/learning/experiments.sqlite").as_posix()}?mode=ro',uri=True)
    try:
        rows=db.execute("""SELECT json_extract(payload,'$.learning.prediction_error'),json_extract(payload,'$.learning.prior_macro.prediction_error')
          FROM steps WHERE run=? AND json_extract(payload,'$.phase')='internal'""",(run,)).fetchall()
    finally:db.close()
    values=np.array([abs(v) for row in rows for v in row if v is not None],float)
    return values[np.isfinite(values)&(values>0)]

def job(spec):
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from learning_brain import LearningBrain,Dynamics
    from learning_adapter import SensoryEncoder,NeuralReadout,ActionCatalog
    from offline_replay import teacher_label
    torch.set_num_threads(1)
    kind,rule,rate,episodes,schema,actor_path,errors=spec
    brain=LearningBrain(dynamics=Dynamics(learning_rate=rate if rate else 1e-4));encoder=SensoryEncoder(ROOT/schema)
    readout=NeuralReadout();catalog=ActionCatalog();dn=readout.indices;brain.set_plastic_posts(dn)
    state=torch.load(actor_path,map_location='cpu',weights_only=False)['action_net']
    weight=state['weight'].numpy().astype(np.float64);bias=state['bias'].numpy().astype(np.float64)
    def policy(features,mask):
        logits=weight@features+bias;logits[~mask]=-np.inf;logits-=logits[mask].max()
        p=np.exp(logits);p/=p.sum();return p
    def replay(plastic):
        brain.plasticity=plastic;rng=np.random.default_rng(12345);out=[];updates=[]
        for number,episode in enumerate(episodes):
            steps=episode['steps'];brain.reset(1000+number);brain.plasticity=plastic;readout.reset();pending=None
            ids,rates,_=encoder.encode(steps[0]['observation']);features=readout.update(brain.step(ids,rates,50.)[0])
            for index in range(1,len(steps)):
                step=steps[index]
                if step['decision_required'] and step['mask'] is not None:
                    mask=np.asarray(step['mask'],bool);label=teacher_label(catalog,steps[index-1]['observation'],mask,index)
                    p=policy(features,mask);out.append(math.log(max(p[label],1e-300)))
                    if plastic:
                        if pending is not None:updates.append(credit(*pending))
                        pending=(brain.capture_motor_eligibility(dn),features.copy(),mask,label,float(rng.choice(errors)))
                ids,rates,_=encoder.encode(step['observation']);features=readout.update(brain.step(ids,rates,50.)[0])
            if plastic and pending is not None:updates.append(credit(*pending))
        return np.array(out),updates
    def credit(snapshot,features,mask,label,error):
        p=policy(features,mask);gradient=weight[label]-p@weight
        c=gradient.reshape(3,len(dn)).sum(0).astype(np.float32);scale=float(np.percentile(np.abs(c),95))
        c=np.clip(c/max(scale,1e-8),-1.,1.).astype(np.float32) if scale>0 else np.zeros_like(c)
        result=brain.reward_motor(snapshot,error,dn,c,sign_aware=rule=='v2');return result['changed_edges']
    if kind=='baseline':
        brain.gains.fill(1.);logp,_=replay(False);return {'kind':kind,'logp':logp.tolist()}
    brain.gains.fill(1.);_,updates=replay(True)
    motor=np.zeros(brain.n,bool);motor[dn]=True;cone=motor[np.asarray(brain.indices,dtype=np.int64)]
    gains=np.asarray(brain.gains[cone],np.float64)
    stats={'mean_abs_gain_change':float(np.mean(np.abs(gains-1))),'max_abs_gain_change':float(np.max(np.abs(gains-1))),
           'edges_changed':int(np.count_nonzero(gains!=1)),'fraction_at_bounds':float(np.mean((gains<=brain.dynamics.gain_min)|(gains>=brain.dynamics.gain_max))),
           'updates':len(updates)}
    logp,_=replay(False)
    return {'kind':kind,'rule':rule,'learning_rate':rate,'logp':logp.tolist(),'gains':stats}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--schema',default='data/learning/sensory_ports_v32.json')
    parser.add_argument('--actor',default='work/probes/offline_actor_v32.pt');parser.add_argument('--episodes',type=int,default=6)
    parser.add_argument('--rates',type=float,nargs='+',default=[1e-4,1e-3,1e-2]);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();started=time.time()
    from offline_replay import load_episodes
    episodes=load_episodes('pilot-v3.1-curriculum','move')
    # Deterministic subset: episodes with the most decisions, trimmed for runtime.
    episodes=sorted(episodes,key=lambda e:(-sum(s['decision_required'] for s in e['steps']),e['trajectory_sha256']))[:args.episodes]
    errors=recorded_prediction_errors()
    specs=[('baseline',None,None,episodes,args.schema,str(ROOT/args.actor),errors)]
    specs+=[('plastic',rule,rate,episodes,args.schema,str(ROOT/args.actor),errors) for rate in args.rates for rule in ('v1','v2')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(job,specs))
    base=np.array(results[0]['logp']);rows=[]
    for r in results[1:]:
        delta=np.array(r['logp'])-base
        rows.append({'rule':'causal_motor_rstdp_'+r['rule'],'learning_rate':r['learning_rate'],'decisions':int(len(delta)),
                     'mean_delta_log_pi_credited':float(delta.mean()),'fraction_improved':float(np.mean(delta>0)),**r['gains']})
    result={'version':1,'schema':args.schema,'actor':args.actor,'episodes':[e['trajectory_sha256'] for e in episodes],
            'ticks':int(sum(len(e['steps']) for e in episodes)),'recorded_prediction_error':{'samples':int(len(errors)),'median_abs':float(np.median(errors)),'p90_abs':float(np.percentile(errors,90))},
            'results':rows,'criterion':'choose the smallest rate whose v2 mean change per episode is 1e-3..1e-2, <1% of motor-cone gains at bounds, and mean delta log pi(credited) > 0',
            'method':__doc__.strip(),'wall_seconds':time.time()-started}
    for row in rows:row['mean_abs_gain_change_per_episode']=row['mean_abs_gain_change']/len(episodes)
    eligible=[row for row in rows if row['rule'].endswith('v2') and 1e-3<=row['mean_abs_gain_change_per_episode']<=1e-2
              and row['fraction_at_bounds']<.01 and row['mean_delta_log_pi_credited']>0]
    result['selected_learning_rate']=min((row['learning_rate'] for row in eligible),default=None)
    from lab_store import atomic_json
    atomic_json(ROOT/'outputs/motor_plasticity_calibration_v32.json',result)
    print(json.dumps({k:result[k] for k in ('ticks','recorded_prediction_error','results','selected_learning_rate','wall_seconds')},indent=2))

if __name__=='__main__':main()
