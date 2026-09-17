from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv,ExerciseTeacher
from learning_policy import make_model,decision,imitate,actor_hash,internal_update
from learning_checkpoint import save_checkpoint,load_checkpoint
from lab_store import Store

def main():
    started=time.time();cfg=json.loads((ROOT/'config/learning.json').read_text());cfg['ppo'].update(n_steps=16,batch_size=8,n_epochs=2)
    brain=LearningBrain();env=FlyOperatorEnv(brain,condition='adapter',stage='switch');env.config=cfg
    model=make_model(env,cfg,7);examples=[];results=[];store=Store('integration-validation-v2')
    for seed in (7,19):
        x,_=env.reset(seed=seed);teacher=ExerciseTeacher(env.catalog)
        for step in range(60):
            action=teacher.choose(env);examples.append((x.copy(),action,env.action_masks().copy()));x,r,t,u,info=env.step(action)
            if t or u:break
        assert info['success'];results.append({'seed':seed,'guided_success':info['success'],'steps':step+1,'controller':'instructor'})
    before=actor_hash(model);metrics=imitate(model,examples,5);after=actor_hash(model);assert before!=after
    env.stage='move';x,_=env.reset();buf=model.rollout_buffer;buf.reset();start=True
    for i in range(16):
        mask=env.action_masks();a,v,lp,detail=decision(model,x,mask,env.catalog,env.readout.body_ids)
        old=x.copy();x,r,t,u,info=env.step(a);assert not (t or u)
        buf.add(old[None,:],np.array([[a]]),np.array([r]),np.array([start]),v,lp,action_masks=mask[None,:]);start=False
    with torch.no_grad():last=model.policy.predict_values(torch.as_tensor(x[None,:]))
    buf.compute_returns_and_advantage(last_values=last,dones=np.array([False]));before_ppo=actor_hash(model);model.train();assert actor_hash(model)!=before_ppo
    before_internal=actor_hash(model);env.train_plasticity=True;previous=x.copy();x,r,t,u,info=env.step(0)
    update=internal_update(model,brain,previous,x,r+.1,t,.999);assert actor_hash(model)==before_internal and update['changed_edges']>0
    p=save_checkpoint('integration_v2',brain,model,env,{'steps':len(examples)+17},store)
    expected_hash=actor_hash(model);expected_rng=torch.rand(8);expected_filters=env.readout.filters.copy()
    with torch.no_grad():model.policy.action_net.weight.add_(1.)
    load_checkpoint(p,brain,model,env);assert actor_hash(model)==expected_hash and torch.equal(torch.rand(8),expected_rng) and np.array_equal(env.readout.filters,expected_filters)
    report={'passed':True,'full_graph':[brain.n,brain.edges],'guided_episodes':results,'imitation':metrics,'actor_changed_after_imitation':True,
      'actor_changed_after_ppo':True,'actor_frozen_during_internal_update':True,'internal_update':update,'policy_rng_filters_checkpoint_exact':True,
      'checkpoint':str(p),'wall_seconds':time.time()-started,'does_not_demonstrate_heldout_skill_learning':True}
    (ROOT/'outputs/training_validation_v2.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);env.close();store.close()
if __name__=='__main__':main()
