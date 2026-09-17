"""Protocol 3.3 PPO for the linear neuronal actor.

Replaces the per-episode MaskablePPO.train() call of protocol 3.2:
- rollouts span many episodes and are measured in cognitive decisions;
- GAE is computed per episode (never continued across an episode boundary) and only
  truncated episodes bootstrap V(s_final);
- an identity test (stored features and masks give ratio 1, KL 0) gates every update;
- actor and critic keep their own learning rates in named optimizer groups. SB3's train()
  is never called, so its _update_learning_rate cannot overwrite them;
- a masked cross-entropy anchor toward the instructor's labels on visited states;
- the critic warms up with the actor frozen before the actor learning rate ramps up.
"""
import math
import numpy as np
import torch

GROUPS=('actor','critic')


def actor_parameters(policy):
    return list(policy.action_net.parameters())+list(policy.mlp_extractor.policy_net.parameters())


def critic_parameters(policy):
    return list(policy.value_net.parameters())+list(policy.mlp_extractor.value_net.parameters())


def install_group_optimizer(model,actor_lr,critic_lr):
    """Fresh Adam with named actor/critic groups covering every trainable policy parameter."""
    policy=model.policy
    groups=[{'params':actor_parameters(policy),'lr':float(actor_lr),'name':'actor'},
            {'params':critic_parameters(policy),'lr':float(critic_lr),'name':'critic'}]
    covered={id(p) for group in groups for p in group['params']}
    missing=[name for name,p in policy.named_parameters() if p.requires_grad and id(p) not in covered]
    if missing:raise ValueError(f'Parameters outside actor/critic groups: {missing}')
    policy.optimizer=torch.optim.Adam(groups,eps=1e-5)
    return policy.optimizer


def group_learning_rates(optimizer):
    return {group.get('name',str(i)):float(group['lr']) for i,group in enumerate(optimizer.param_groups)}


def set_group_learning_rates(optimizer,**rates):
    names={group.get('name') for group in optimizer.param_groups}
    if set(rates)-names:raise ValueError(f'Unknown optimizer groups: {sorted(set(rates)-names)}')
    for group in optimizer.param_groups:
        if group.get('name') in rates:group['lr']=float(rates[group['name']])


def parameter_vector(parameters):
    return torch.cat([p.detach().reshape(-1).cpu() for p in parameters]) if parameters else torch.zeros(0)


def episode_advantages(rewards,values,macro_steps,gamma,gae_lambda,bootstrap_value=None):
    """GAE inside one episode. bootstrap_value is None for a terminal episode.

    A decision that lasted k engine ticks discounts by gamma**k; lambda applies once per decision.
    """
    n=len(rewards);advantages=np.zeros(n,np.float64);running=0.
    next_value=0. if bootstrap_value is None else float(bootstrap_value)
    for i in range(n-1,-1,-1):
        discount=gamma**int(macro_steps[i])
        delta=float(rewards[i])+discount*next_value-float(values[i])
        running=delta+discount*gae_lambda*running
        advantages[i]=running;next_value=float(values[i])
    returns=advantages+np.asarray(values,np.float64)
    return advantages.astype(np.float32),returns.astype(np.float32)


class RolloutCollector:
    """Cognitive decisions from complete episodes of one learner (one condition and seed)."""
    def __init__(self):self.episodes=[];self.excluded=0
    @property
    def decisions(self):return sum(len(e['transitions']) for e in self.episodes)
    def add_episode(self,transitions,terminated,truncated,bootstrap_value=None,excluded=False,episode_uid=None):
        if excluded:self.excluded+=1;return False
        if not (terminated or truncated):raise ValueError('Rollouts accept complete episodes only')
        if not terminated and bootstrap_value is None:raise ValueError('A truncated episode needs V(s_final) for its bootstrap')
        if not transitions:return False
        self.episodes.append({'transitions':transitions,'terminated':bool(terminated),'truncated':bool(truncated and not terminated),
                              'bootstrap_value':None if terminated else float(bootstrap_value),'episode_uid':episode_uid})
        return True
    def clear(self):self.episodes=[];self.excluded=0
    def batch(self,gamma,gae_lambda):
        observations=[];actions=[];masks=[];old_logprob=[];old_values=[];advantages=[];returns=[];teacher=[]
        for episode in self.episodes:
            items=episode['transitions']
            adv,ret=episode_advantages([t['reward'] for t in items],[t['value'] for t in items],[t.get('macro_steps',1) for t in items],
                                       gamma,gae_lambda,episode['bootstrap_value'])
            advantages.append(adv);returns.append(ret)
            for t in items:
                observations.append(np.asarray(t['observation'],np.float32));actions.append(int(t['action']));masks.append(np.asarray(t['mask'],bool))
                old_logprob.append(float(t['logprob']));old_values.append(float(t['value']))
                label=t.get('teacher');teacher.append(int(label) if label is not None and bool(t['mask'][int(label)]) else -1)
        if not observations:raise ValueError('Empty rollout')
        return {'observations':torch.from_numpy(np.stack(observations)),'actions':torch.tensor(actions,dtype=torch.long),
                'masks':np.stack(masks),'old_logprob':torch.tensor(old_logprob,dtype=torch.float32),
                'old_values':torch.tensor(old_values,dtype=torch.float32),'advantages':torch.from_numpy(np.concatenate(advantages)),
                'returns':torch.from_numpy(np.concatenate(returns)),'teacher':torch.tensor(teacher,dtype=torch.long),
                'episodes':len(self.episodes)}


def identity_check(policy,batch,tolerance_ratio=1e-5,tolerance_kl=1e-6):
    """Recompute log-probabilities from stored features and masks with the unchanged policy."""
    with torch.no_grad():
        _,logprob,_=policy.evaluate_actions(batch['observations'],batch['actions'],action_masks=batch['masks'])
        log_ratio=logprob-batch['old_logprob'];ratio=torch.exp(log_ratio)
        kl=float(((ratio-1)-log_ratio).mean());deviation=float((ratio-1).abs().max())
    return {'max_ratio_deviation':deviation,'approx_kl':kl,'passed':bool(deviation<=tolerance_ratio and kl<=tolerance_kl)}


def anchor_beta(decisions,settings):
    """Anchor weight decays by accumulated decisions (half-life) down to a floor."""
    beta0=float(settings.get('anchor_beta',1.));half_life=float(settings.get('anchor_half_life_decisions',20000))
    return max(float(settings.get('anchor_beta_floor',.05)),beta0*0.5**(decisions/half_life)) if beta0>0 else 0.


def new_state():
    return {'phase':'critic_warmup','rollouts':0,'updates':0,'actor_updates':0,'critic_warmup_updates':0,'decisions':0,
            'warmup_mse':[],'identity_failures':0,'skipped_rollouts':0,'history':[]}


def learning_rates_for(state,settings):
    if state['phase']=='critic_warmup':
        return {'actor':0.,'critic':float(settings.get('critic_warmup_learning_rate',settings['critic_learning_rate']))}
    ramp=max(1,int(settings.get('actor_ramp_updates',4)))
    return {'actor':float(settings['actor_learning_rate'])*min(1.,(state['actor_updates']+1)/ramp),'critic':float(settings['critic_learning_rate'])}


def ppo_update(model,collector,settings,state):
    """One update from a complete rollout. Mutates state (JSON-serializable) and returns diagnostics."""
    policy=model.policy;optimizer=policy.optimizer
    state['rollouts']+=1;decisions=collector.decisions;state['decisions']+=decisions
    state['stage_decisions']=state.get('stage_decisions',0)+decisions
    batch=collector.batch(float(settings['gamma']),float(settings['gae_lambda']))
    n=len(batch['actions']);diagnostics={'rollout':state['rollouts'],'decisions':decisions,'episodes':batch['episodes'],
      'excluded_episodes':collector.excluded,'phase':state['phase']}
    raw_adv=batch['advantages']
    diagnostics['advantage']={'mean':float(raw_adv.mean()),'std':float(raw_adv.std()) if n>1 else 0.,'min':float(raw_adv.min()),'max':float(raw_adv.max())}
    identity=identity_check(policy,batch,float(settings.get('identity_ratio_tolerance',1e-5)),float(settings.get('identity_kl_tolerance',1e-6)))
    diagnostics['identity']=identity
    if not identity['passed']:
        state['identity_failures']+=1;diagnostics.update(updated=False,reason='identity_check_failed')
        _remember(state,diagnostics);return diagnostics
    every=max(1,int(settings.get('update_every_rollouts',1)))
    if state['rollouts']%every:
        state['skipped_rollouts']+=1;diagnostics.update(updated=False,reason='update_frequency_control')
        _remember(state,diagnostics);return diagnostics
    value_mse_before=float(((batch['old_values']-batch['returns'])**2).mean())
    variance=float(batch['returns'].var()) if n>1 else 0.
    diagnostics['value_mse_before']=value_mse_before
    diagnostics['explained_variance_before']=float(1-((batch['returns']-batch['old_values']).var()/variance)) if variance>1e-12 else None
    warmup=state['phase']=='critic_warmup'
    if warmup:
        state['warmup_mse'].append(value_mse_before)
    rates=learning_rates_for(state,settings);set_group_learning_rates(optimizer,**rates);diagnostics['learning_rates']=rates
    advantages=batch['advantages']
    mode=settings.get('advantage_normalization','rollout')
    if mode=='rollout' and n>1:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    elif mode!='none' and mode!='rollout':raise ValueError(f'Unknown advantage normalization: {mode}')
    # The anchor decays with decisions at the current stage: a new skill starts fully anchored.
    beta=0. if warmup else anchor_beta(state['stage_decisions'],settings);diagnostics['anchor_beta']=beta
    actor_before=parameter_vector(actor_parameters(policy));critic_before=parameter_vector(critic_parameters(policy))
    # The warm-up fits the critic alone, so it may use more passes and smaller minibatches.
    epochs=int(settings.get('critic_warmup_epochs' if warmup else 'epochs',settings.get('epochs',2)))
    minibatch=max(2,int(settings.get('critic_warmup_minibatch_size' if warmup else 'minibatch_size',settings.get('minibatch_size',256))))
    clip=float(settings.get('clip_range',.15));target_kl=settings.get('target_kl');vf_coef=float(settings.get('vf_coef',.5))
    ent_coef=float(settings.get('ent_coef',0.));max_grad=float(settings.get('max_grad_norm',.5))
    losses={'policy':[],'value':[],'anchor':[],'entropy':[],'approx_kl':[],'clip_fraction':[],'grad_norm':[]}
    stopped=False;completed_epochs=0;steps=0
    for epoch in range(epochs):
        order=torch.randperm(n)
        # Minibatches never shrink below half the requested size: a tiny tail is merged.
        starts=list(range(0,n,minibatch))
        if len(starts)>1 and n-starts[-1]<minibatch//2:starts.pop()
        for k,start in enumerate(starts):
            end=n if k==len(starts)-1 else start+minibatch;index=order[start:end]
            obs=batch['observations'][index];masks=batch['masks'][index.numpy()]
            if warmup:
                values=policy.predict_values(obs).flatten()
                value_loss=((values-batch['returns'][index])**2).mean();loss=vf_coef*value_loss
                losses['value'].append(float(value_loss.detach()))
            else:
                values,logprob,entropy=policy.evaluate_actions(obs,batch['actions'][index],action_masks=masks)
                values=values.flatten();log_ratio=logprob-batch['old_logprob'][index];ratio=torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl=float(((ratio-1)-log_ratio).mean());clip_fraction=float(((ratio-1).abs()>clip).float().mean())
                if target_kl is not None and approx_kl>1.5*float(target_kl):
                    stopped=True;losses['approx_kl'].append(approx_kl);break
                adv=advantages[index]
                policy_loss=-torch.min(adv*ratio,adv*torch.clamp(ratio,1-clip,1+clip)).mean()
                value_loss=((values-batch['returns'][index])**2).mean()
                entropy_loss=-(entropy.mean() if entropy is not None else -logprob.mean())
                loss=policy_loss+vf_coef*value_loss+ent_coef*entropy_loss
                teacher=batch['teacher'][index];labelled=teacher>=0
                if beta>0 and bool(labelled.any()):
                    distribution=policy.get_distribution(obs[labelled],action_masks=masks[labelled.numpy()])
                    anchor=-distribution.log_prob(teacher[labelled]).mean();loss=loss+beta*anchor
                    losses['anchor'].append(float(anchor.detach()))
                losses['policy'].append(float(policy_loss.detach()));losses['value'].append(float(value_loss.detach()))
                losses['entropy'].append(float(-entropy_loss.detach()));losses['approx_kl'].append(approx_kl);losses['clip_fraction'].append(clip_fraction)
            optimizer.zero_grad();loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),max_grad);losses['grad_norm'].append(float(grad_norm))
            optimizer.step();steps+=1
        if stopped:break
        completed_epochs+=1
    actor_after=parameter_vector(actor_parameters(policy));critic_after=parameter_vector(critic_parameters(policy))
    for p in policy.parameters():
        if not torch.isfinite(p).all():raise FloatingPointError('PPO produced non-finite policy parameters')
    actor_delta=float((actor_after-actor_before).norm());critic_delta=float((critic_after-critic_before).norm())
    if warmup and actor_delta!=0.:raise AssertionError('Actor changed during critic warm-up')
    state['updates']+=1
    if warmup:state['critic_warmup_updates']+=1
    else:state['actor_updates']+=1
    mean=lambda values:float(np.mean(values)) if values else None
    diagnostics.update(updated=True,epochs_completed=completed_epochs,optimizer_steps=steps,stopped_by_target_kl=stopped,
      policy_loss=mean(losses['policy']),value_loss=mean(losses['value']),anchor_loss=mean(losses['anchor']),entropy=mean(losses['entropy']),
      approx_kl=mean(losses['approx_kl']),approx_kl_last=losses['approx_kl'][-1] if losses['approx_kl'] else None,
      clip_fraction=mean(losses['clip_fraction']),grad_norm=mean(losses['grad_norm']),actor_delta_norm=actor_delta,critic_delta_norm=critic_delta,
      labelled_fraction=float((batch['teacher']>=0).float().mean()))
    if warmup:_maybe_finish_warmup(state,settings,diagnostics)
    _remember(state,diagnostics)
    return diagnostics


def _maybe_finish_warmup(state,settings,diagnostics):
    """End the critic warm-up once its error on each new (unseen) rollout stops decreasing."""
    history=state['warmup_mse'];minimum=int(settings.get('critic_warmup_min_rollouts',2));maximum=int(settings.get('critic_warmup_max_rollouts',6))
    plateau=len(history)>=2 and history[-1]>=float(settings.get('critic_warmup_improvement',.95))*min(history[:-1])
    if len(history)>=maximum or (len(history)>=minimum and plateau):
        state['phase']='actor';diagnostics['critic_warmup_finished']={'rollouts':len(history),'reason':'maximum' if len(history)>=maximum and not plateau else 'plateau'}


def _remember(state,diagnostics,keep=40):
    state['history']=(state['history']+[diagnostics])[-keep:]
