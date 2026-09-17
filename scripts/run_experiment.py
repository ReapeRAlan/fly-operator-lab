"""Closed loop with the real game and every classified MaleCNS v1.0 neuron."""
from pathlib import Path
import sys,json,time,collections,argparse,hashlib,platform
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge
from flybrain import FlyBrain,Adapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=20);p.add_argument('--seconds',type=int,default=3)
    p.add_argument('--frames',type=int,help='Exact step/frame count per episode, overriding seconds')
    p.add_argument('--last-dt',type=int,help='Optional final step duration for exact simulated-time exports')
    p.add_argument('--record',action='store_true');p.add_argument('--interventions',action='store_true');p.add_argument('--name',default='live_validation')
    args=p.parse_args(); assert args.episodes>0 and args.seconds>0
    out=ROOT/'outputs'/args.name;out.mkdir(exist_ok=False)
    brain=FlyBrain();adapter=Adapter();summary={'dataset':'male-cns:v1.0','neurons':len(brain.ids),'edges':len(brain.indices),'session':json.loads((ROOT/'work/session.json').read_text()),'python':platform.python_version(),'arguments':vars(args),'episodes':[]}
    summary['adapter_sha256']=hashlib.sha256((ROOT/'config/adapter.json').read_bytes()).hexdigest()
    started=time.perf_counter();errors=[]
    with Bridge() as b, (out/'trace.jsonl').open('w',encoding='utf-8') as trace:
        try:
            for index in range(args.episodes):
                mode=['normal','altered_inputs','disconnected_outputs'][index] if args.interventions and index<3 else 'normal'
                seed=7 if index<3 else 7+index
                brain.reset(seed);o=b.new_episode(record=args.record);initial=o;wall=time.perf_counter();actions=collections.Counter();spikes_total=0;frames=0
                step_count=args.frames or args.seconds*(30 if args.record else 20)
                reason='horizon'
                for step in range(step_count):
                    duration=[33,33,34][step%3] if args.record else 50
                    if step==step_count-1 and args.last_dt is not None:duration=args.last_dt
                    ids,rates,channels=adapter.encode(o)
                    if mode=='altered_inputs': rates=rates[::-1].copy()
                    spikes,activity=brain.step(ids,rates,duration)
                    proposed,scores=adapter.decode(spikes,o,duration)
                    command=proposed if mode!='disconnected_outputs' else {'action':'stop'}
                    prev=o
                    o=b.step(command,duration)
                    if o['sim_time_ms']-prev['sim_time_ms']!=duration:raise RuntimeError('Game simulation time differs from neural step')
                    if o['sequence']!=prev['sequence']+1:raise RuntimeError('Duplicate or missing game command')
                    if args.record:
                        if o['frame']['client_sim_time_ms']!=o['sim_time_ms']:raise RuntimeError('Captured client frame not synchronized with server')
                        frames+=1
                    actions[command['action']]+=1;spikes_total+=activity['spikes']
                    trace.write(json.dumps({'episode_index':index,'mode':mode,'seed':seed,'wall_seconds':time.perf_counter()-started,'observation':prev,'channels':channels,'activity':activity,'spike_counts_sha256':hashlib.sha256(spikes.tobytes()).hexdigest(),'scores':scores,'proposed_command':proposed,'executed_command':command,'result':o})+'\n')
                    trace.flush()
                    if step%100==0:print(f'EP {index+1}/{args.episodes} {mode} sim={o["sim_time_ms"]}ms actions={dict(actions)}',flush=True)
                    if o['game_state']==2 or o.get('operator',{}).get('health',0)<=0:reason='terminal';break
                item={'index':index,'mode':mode,'seed':seed,'game_episode':o['episode'],'reason':reason,'steps':o['sequence'],'sim_time_ms':o['sim_time_ms'],'wall_seconds':time.perf_counter()-wall,'spikes':spikes_total,'actions':dict(actions),'initial':initial,'final':o,'frames':frames,'rss_mib':psutil.Process().memory_info().rss/2**20}
                summary['episodes'].append(item)
                (out/'summary.json').write_text(json.dumps(summary,indent=2))
                print(f'COMPLETE {index+1}: {item["sim_time_ms"]}ms, {spikes_total} spikes, {dict(actions)}',flush=True)
        except BaseException as e:
            errors.append(repr(e));raise
        finally:
            summary['errors']=errors;summary['wall_seconds']=time.perf_counter()-started
            summary['completed_episodes']=len(summary['episodes']);summary['passed']=not errors and len(summary['episodes'])==args.episodes
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SAVED',out,flush=True)
if __name__=='__main__':main()
