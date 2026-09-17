"""Linear neuronal actor and separate value estimator; no raw game input to either."""
import hashlib,io
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

def make_model(env,config,seed):
    torch.set_num_threads(config.get('max_cpu_threads',2))
    model=MaskablePPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,
      policy_kwargs={'net_arch':dict(pi=[],vf=[64,64]),'activation_fn':torch.nn.Tanh},**config['ppo'])
    model.set_logger(configure(None,[]))
    if 'ppo_v33' in config:
        # Protocol 3.3: named actor/critic groups; imitation never touches this optimizer.
        from learning_ppo import install_group_optimizer
        install_group_optimizer(model,0.,config['ppo_v33']['critic_learning_rate'])
    return model

def actor_hash(model):
    h=hashlib.sha256()
    for value in model.policy.action_net.state_dict().values():h.update(memoryview(np.ascontiguousarray(value.detach().cpu().numpy())).cast('B'))
    return h.hexdigest()

def decision(model,features,mask,catalog,body_ids,deterministic=False):
    x=torch.as_tensor(features[None,:],dtype=torch.float32)
    with torch.no_grad():
        distribution=model.policy.get_distribution(x,action_masks=mask[None,:])
        actions=distribution.get_actions(deterministic=deterministic)
        action=int(actions[0]);value=model.policy.predict_values(x)
        logprob=distribution.log_prob(actions);probabilities=distribution.distribution.probs[0].numpy()
        entropy=float(distribution.distribution.entropy()[0])
        weights=model.policy.action_net.weight[action].numpy();bias=float(model.policy.action_net.bias[action])
    contribution=weights*features;order=np.argsort(np.abs(contribution))[-20:][::-1]
    detail={'selected':action,'probabilities':probabilities.tolist(),'action_labels':[catalog.label(i) for i in range(len(mask))],
      'mask':mask.tolist(),'value':float(value[0]),'bias':bias,'logit':float(bias+contribution.sum()),
      'contribution_sum':float(contribution.sum()),'entropy':entropy,'wait_probability':float(probabilities[0]),
      'contributions':[{'body_id':int(body_ids[i%len(body_ids)]),
      'filter_ms':[100,500,2000][i//len(body_ids)],'weight':float(weights[i]),'feature':float(features[i]),'logit_contribution':float(contribution[i])} for i in order],
      'meaning':'Exact contributions to the selected linear actor logit. Association is not a causal intervention.'}
    return action,value,logprob,detail

def angular_neighbours(catalog):
    """Adjacent directions within one directional action family (move, turn, throw per slot)."""
    families={}
    for i,e in enumerate(catalog.entries):
        if 'angle' in e:families.setdefault((e['action'],e.get('slot')),[]).append(i)
    neighbours={}
    for members in families.values():
        angles=np.array([catalog.entries[i]['angle'] for i in members],float)
        for i in members:
            distance=np.abs((angles-catalog.entries[i]['angle']+np.pi)%(2*np.pi)-np.pi);positive=distance[distance>1e-9]
            if len(positive):neighbours[i]=[members[j] for j in np.flatnonzero(np.abs(distance-positive.min())<1e-6)]
    return neighbours

def _imitate_v32(model,examples,settings,catalog):
    """Fit the linear actor with its own optimizer: episode-held-out early stopping, then refit on all data.

    PPO's optimizer state is untouched. Adjacent directions receive label mass because the
    instructor's nearest-of-eight choice makes neighbouring moves nearly equivalent.
    """
    net=model.policy.action_net;params=list(net.parameters())
    x=torch.from_numpy(np.stack([e[0] for e in examples]));y=torch.tensor([int(e[1]) for e in examples])
    masks=torch.from_numpy(np.stack([e[2] for e in examples]));stages=[e[3] if len(e)>3 else 'unknown' for e in examples]
    groups={}
    for stage,action in zip(stages,y.tolist()):groups[(stage,action)]=groups.get((stage,action),0)+1
    weights=torch.tensor([1/groups[(stage,action)] for stage,action in zip(stages,y.tolist())],dtype=torch.float32)
    weights*=len(weights)/weights.sum()
    neighbours=angular_neighbours(catalog) if catalog is not None else {}
    smoothing=float(settings.get('angular_label_smoothing',0.));targets=torch.zeros(masks.shape,dtype=torch.float32)
    for row,label in enumerate(y.tolist()):
        valid=[j for j in neighbours.get(label,[]) if bool(masks[row,j])]
        if smoothing>0 and valid:
            targets[row,label]=1-smoothing
            for j in valid:targets[row,j]=smoothing/len(valid)
        else:targets[row,label]=1.
    tolerated=[{label,*neighbours.get(label,[])} for label in y.tolist()]
    # Without recorded episode ids, contiguous blocks of 16 decisions stand in for episodes.
    episodes=[e[4] if len(e)>4 else index//16 for index,e in enumerate(examples)]
    unique=sorted(set(episodes));period=max(2,round(1/max(1e-6,float(settings.get('validation_fraction',.2)))))
    held={g for position,g in enumerate(unique) if position%period==period-1}
    validation=torch.tensor([i for i,g in enumerate(episodes) if g in held],dtype=torch.long)
    train=torch.tensor([i for i,g in enumerate(episodes) if g not in held],dtype=torch.long)
    initial=[p.detach().clone() for p in params];batch=int(settings.get('batch_size',64))
    def loss_of(index):
        logits=net(x[index]).masked_fill(~masks[index],-1e8)
        per=-(targets[index]*torch.log_softmax(logits,1)).sum(1)
        return (per*weights[index]).sum()/weights[index].sum().clamp(min=1e-8),logits
    def fit(index,epochs,monitor=None):
        with torch.no_grad():
            for p,v in zip(params,initial):p.copy_(v)
        optimizer=torch.optim.Adam(params,lr=float(settings['learning_rate']),weight_decay=float(settings.get('weight_decay',0.)))
        history=[];best=(float('inf'),0);stale=0
        for epoch in range(epochs):
            order=index[torch.randperm(len(index))]
            for start in range(0,len(order),batch):
                loss,_=loss_of(order[start:start+batch]);optimizer.zero_grad();loss.backward();optimizer.step()
            if monitor is not None:
                with torch.no_grad():
                    value,logits=loss_of(monitor);prediction=logits.argmax(1)
                    exact=float((prediction==y[monitor]).float().mean())
                    within=float(np.mean([int(p) in tolerated[int(i)] for p,i in zip(prediction,monitor)]))
                history.append({'epoch':epoch+1,'loss':float(value),'exact':exact,'within_tolerance':within})
                if float(value)<best[0]-1e-6:best=(float(value),epoch+1);stale=0
                else:
                    stale+=1
                    if stale>=int(settings.get('patience',30)):break
        return history,best
    selected=None
    if len(validation) and len(train)>=2:
        history,best=fit(train,int(settings.get('max_epochs',300)),validation);selected=history[best[1]-1] if best[1] else None
        epochs=max(int(settings.get('min_epochs',5)),best[1])
    else:epochs=int(settings.get('min_epochs',20))
    fit(torch.arange(len(y)),epochs)
    with torch.no_grad():
        logits=net(x).masked_fill(~masks,-1e8);prediction=logits.argmax(1)
    stage_counts={stage:stages.count(stage) for stage in sorted(set(stages))}
    return {'version':'3.2','examples':len(y),'epochs':epochs,'class_counts':torch.bincount(y,minlength=masks.shape[1]).tolist(),
      'stage_counts':stage_counts,'accuracy':float((prediction==y).float().mean()),
      'within_tolerance':float(np.mean([int(p) in tolerated[i] for i,p in enumerate(prediction.tolist())])),
      'validation_episodes':len(held),'validation_examples':len(validation),'validation_at_selected_epoch':selected,
      'training_prediction_wait_fraction':float((prediction==0).float().mean()),'settings':settings}

def imitate(model,examples,epochs=20,settings=None,catalog=None,own_optimizer=True,learning_rate=3e-4):
    """Behaviour cloning of the linear actor.

    own_optimizer (protocol 3.3, the default) fits action_net with a fresh Adam per call, so
    PPO's optimizer moments never mix with imitation gradients and the actor group's PPO
    learning rate (zero during the critic warm-up) cannot silence cloning.
    """
    if not examples:return {}
    if settings is not None:return _imitate_v32(model,examples,settings,catalog)
    optimizer=torch.optim.Adam(model.policy.action_net.parameters(),lr=learning_rate) if own_optimizer else model.policy.optimizer
    clipped=list(model.policy.action_net.parameters()) if own_optimizer else list(model.policy.parameters())
    x=torch.from_numpy(np.stack([e[0] for e in examples]));y=torch.tensor([e[1] for e in examples])
    masks=torch.from_numpy(np.stack([e[2] for e in examples]));counts=torch.bincount(y,minlength=masks.shape[1]).float()
    stages=[e[3] if len(e)>3 else 'unknown' for e in examples]
    groups={}
    for stage,action in zip(stages,y.tolist()):groups[(stage,action)]=groups.get((stage,action),0)+1
    sample_weights=torch.tensor([1/groups[(stage,action)] for stage,action in zip(stages,y.tolist())],dtype=torch.float32)
    sample_weights*=len(sample_weights)/sample_weights.sum();losses=[]
    for epoch in range(epochs):
        order=torch.randperm(len(y))
        for start in range(0,len(y),64):
            ix=order[start:start+64];logits=model.policy.action_net(x[ix]).masked_fill(~masks[ix],-1e8)
            per_example=torch.nn.functional.cross_entropy(logits,y[ix],reduction='none')
            loss=(per_example*sample_weights[ix]).sum()/sample_weights[ix].sum().clamp(min=1e-8)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(clipped,.5);optimizer.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        logits=model.policy.action_net(x).masked_fill(~masks,-1e8);prediction=logits.argmax(1)
        accuracy=float((prediction==y).float().mean());nonwait=y!=0
        nonwait_accuracy=float((prediction[nonwait]==y[nonwait]).float().mean()) if nonwait.any() else None
    stage_counts={stage:stages.count(stage) for stage in sorted(set(stages))}
    return {'examples':len(y),'epochs':epochs,'loss_last':losses[-1],'class_counts':counts.int().tolist(),'own_optimizer':bool(own_optimizer),
      'stage_counts':stage_counts,'accuracy':accuracy,'nonwait_accuracy':nonwait_accuracy,
      'training_prediction_wait_fraction':float((prediction==0).float().mean())}

def internal_update(model,brain,previous,next_features,reward,terminated,gamma,reward_gain=1.,
                    eligibility_snapshot=None,action=None,mask=None,descending_indices=None,rule='causal_motor_rstdp_v1'):
    x=torch.as_tensor(previous[None,:],dtype=torch.float32)
    value=model.policy.predict_values(x)[0,0]
    with torch.no_grad():following=model.policy.predict_values(torch.as_tensor(next_features[None,:],dtype=torch.float32))[0,0]
    # External time-limit truncation retains the bootstrap; native terminal does not.
    target=torch.as_tensor(reward)+(0. if terminated else gamma*following)
    delta=float((target-value).detach());modulation=float(np.clip(delta*reward_gain,-1,1))
    if eligibility_snapshot is None:
        update=brain.reward(modulation)
    else:
        if action is None or mask is None or descending_indices is None:raise ValueError('Causal motor plasticity requires action, mask and descending indices')
        with torch.no_grad():
            distribution=model.policy.get_distribution(x,action_masks=np.asarray(mask,bool)[None,:])
            probabilities=distribution.distribution.probs[0]
            weights=model.policy.action_net.weight
            gradient=weights[int(action)]-probabilities@weights
            n=len(descending_indices)
            if gradient.numel()!=3*n:raise ValueError('Descending readout shape mismatch')
            credit=gradient.reshape(3,n).sum(0).detach().cpu().numpy().astype(np.float32)
            scale=float(np.percentile(np.abs(credit),95))
            credit=np.clip(credit/max(scale,1e-8),-1.,1.) if scale>0 else np.zeros_like(credit)
        update=brain.reward_motor(eligibility_snapshot,modulation,descending_indices,credit,sign_aware=rule=='causal_motor_rstdp_v2')
    loss=(value-target.detach()).square()
    model.policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);model.policy.optimizer.step()
    update.update(value_loss=float(loss.detach()),prediction_error=delta,modulation_signal=modulation,reward_gain=float(reward_gain));return update
