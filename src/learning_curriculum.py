"""Curriculum helpers shared by the v3 trainer and focused contract tests."""
from collections import Counter
import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer


def exact_minibatch_size(sample_count,preferred):
    """Choose a minibatch >=2 that divides the rollout exactly.

    Stable-Baselines normalizes advantages inside each minibatch. A trailing
    minibatch of one sample has an undefined sample standard deviation and can
    turn the complete policy into NaNs (for example 65 samples with batch 64).
    """
    sample_count=int(sample_count);preferred=int(preferred)
    if sample_count<2:raise ValueError('PPO needs at least two samples')
    for size in range(min(preferred,sample_count),1,-1):
        if sample_count%size==0:return size
    return sample_count


def run_checkpointed_evaluation(save,restore,evaluate,set_checkpointing_enabled):
    """Run validation as a transaction and always restore the training state.

    Checkpoint callbacks are disabled only after the clean save exists. They
    remain disabled through restore, so a stop, resource pause or exception in
    validation cannot replace the clean checkpoint with evaluation state.
    """
    token=save();set_checkpointing_enabled(False)
    try:return evaluate()
    finally:
        try:restore(token)
        finally:set_checkpointing_enabled(True)


def forced_wait(mask):
    mask=np.asarray(mask,dtype=bool)
    return bool(len(mask) and mask[0] and np.count_nonzero(mask)==1)


def add_demonstration(examples,features,teacher_action,mask,stage,episode=None):
    """Store cognitive choices only; native progress ticks are not wait labels.

    The optional episode id lets imitation hold out whole episodes for early stopping.
    """
    if forced_wait(mask):return False
    item=(np.asarray(features,np.float32).copy(),int(teacher_action),np.asarray(mask,bool).copy(),str(stage))
    examples.append(item if episode is None else item+(int(episode),))
    return True


def rollout_metrics(transitions):
    actions=[int(t['action']) for t in transitions]
    required=[not forced_wait(t['mask']) for t in transitions]
    free=[a for a,r in zip(actions,required) if r]
    return {'ticks':len(transitions),'decisions':len(free),'forced_waits':len(transitions)-len(free),
            'free_waits':sum(a==0 for a in free),'action_counts':dict(Counter(actions)),
            'free_wait_fraction':sum(a==0 for a in free)/len(free) if free else 0.}


def train_adapter_episode(model,transitions,final_features,terminated,truncated,config):
    """One complete natural episode becomes one mask-aware PPO update."""
    if len(transitions)<2:return {**rollout_metrics(transitions),'updated':False,'reason':'episode_too_short'}
    ppo=config['ppo'];rewards=[float(t['reward']) for t in transitions]
    buffer=MaskableRolloutBuffer(len(transitions),model.observation_space,model.action_space,device=model.device,
                                 gamma=ppo['gamma'],gae_lambda=ppo['gae_lambda'],n_envs=1)
    for item,reward in zip(transitions,rewards):
        buffer.add(item['observation'][None,:],np.array([[item['action']]]),np.array([reward],np.float32),
                   np.array([item['episode_start']]),item['value'],item['logprob'],action_masks=item['mask'][None,:])
    with torch.no_grad():
        next_value=0. if terminated else float(model.policy.predict_values(torch.as_tensor(final_features[None,:],dtype=torch.float32))[0])
    advantage=0.;advantages=np.zeros(len(transitions),np.float32);returns=np.zeros(len(transitions),np.float32)
    for i in range(len(transitions)-1,-1,-1):
        item=transitions[i];discount=ppo['gamma']**int(item.get('macro_steps',1))
        nonterminal=0. if i==len(transitions)-1 and terminated else 1.
        value=float(np.asarray(item['value'].detach().cpu()).reshape(-1)[0])
        delta=float(item['reward'])+discount*next_value*nonterminal-value
        advantage=delta+discount*ppo['gae_lambda']*nonterminal*advantage
        advantages[i]=advantage;returns[i]=advantage+value;next_value=value
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():raise FloatingPointError('Non-finite PPO targets')
    buffer.advantages[:,0]=advantages;buffer.returns[:,0]=returns
    old_buffer,old_batch=model.rollout_buffer,model.batch_size
    try:
        model.rollout_buffer=buffer;model.batch_size=exact_minibatch_size(len(transitions),old_batch);model.train()
        parameters=getattr(model.policy,'parameters',None)
        if parameters is not None and not all(torch.isfinite(p).all().item() for p in parameters()):
            raise FloatingPointError('PPO produced non-finite policy parameters')
    finally:model.rollout_buffer=old_buffer;model.batch_size=old_batch
    return {**rollout_metrics(transitions),'updated':True,'bootstrap_applied':bool(truncated and not terminated),
            'environment_ticks':sum(int(t.get('macro_steps',1)) for t in transitions)}
