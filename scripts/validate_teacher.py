"""Validate the training instructor in the native engine, without running a neural model."""
from pathlib import Path
import sys,json,time,types,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge
from learning_env import ExerciseTeacher,FlyOperatorEnv
from learning_adapter import ActionCatalog,clean_observation
from learning_runtime import WorkerLock
class Harness:
    potential=FlyOperatorEnv.potential;skill_success=FlyOperatorEnv.skill_success
    def action_masks(self):return self.catalog.mask(self.obs,self.allowed)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    missions=json.loads((ROOT/'data/learning/scenarios.json').read_text());cfg=json.loads((ROOT/'config/learning.json').read_text());cap=json.loads((ROOT/'outputs/native_capabilities_v2.json').read_text())
    rows=[];start=time.time();scenario_seeds=(7,19,30)
    with Bridge() as bridge:
        for stage in cfg['stage_order']:
            for seed in scenario_seeds:
                e=Harness();e.catalog=ActionCatalog();e.allowed=set(cap['validated_actions']);e.mission=next(m for m in missions if m['stage']==stage and m['seed']==seed and m['split']=='train')
                e.raw=bridge.new_mission(e.mission['name']);e.start_raw=e.raw;e.obs=clean_observation(e.raw,e.mission);e.steps=0;e.shot_ever=False;e.rescue_followed=False;e.cancel_practiced=False;teacher=ExerciseTeacher(e.catalog);trace=[]
                for i in range(e.mission['max_seconds']*20):
                    a=teacher.choose(e);command=e.catalog.decode(a,e.obs);e.raw=bridge.step(command,50);e.steps+=1;e.obs=clean_observation(e.raw,e.mission);e.shot_ever |=e.raw.get('shots_accepted',0)>0;e.rescue_followed |=command['action']=='follow';e.cancel_practiced |=e.raw.get('action_receipt',{}).get('action')=='cancel' and e.raw.get('action_receipt',{}).get('status')=='completed'
                    trace.append({'command':command,'receipt':e.raw['action_receipt'],'position':e.raw['operator']['position'],'visible_ids':e.raw.get('visible_ids',[]),'ammo':e.raw['operator']['ammo']})
                    if e.skill_success(e.raw) or e.raw.get('mission_result') or e.raw.get('game_state')==2:break
                target=next((o for o in e.raw.get('objects',[]) if o.get('id')==e.mission.get('target_id')),None)
                actor=e.raw.get('operator',{})
                row={'stage':stage,'seed':seed,'mission':e.mission['name'],'success':e.skill_success(e.raw),'steps':e.steps,
                     'mission_result':e.raw.get('mission_result'),'final_state':{
                       'position':actor.get('position'),'health':actor.get('health'),'ammo':actor.get('ammo'),
                       'commands':actor.get('commands'),'crouched':actor.get('crouched'),
                       'receipt':e.raw.get('action_receipt'),'target':target},'trace':trace};rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
                inputs={str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in (
                  ROOT/'config/learning.json',ROOT/'data/learning/scenarios.json',ROOT/'outputs/native_capabilities_v2.json',
                  ROOT/'src/learning_adapter.py',ROOT/'src/learning_env.py',ROOT/'work/build/flybridge.dll')}
                artifact={'version':'3.1','scope':'Native instructor rules only; no neural simulation; staged curriculum actions',
                          'session':bridge.session,'scenario_seeds':list(scenario_seeds),'inputs_sha256':inputs,
                          'rows':rows,'passed':len(rows)==len(cfg['stage_order'])*len(scenario_seeds) and all(r['success'] for r in rows),
                          'wall_seconds':time.time()-start}
                (ROOT/'outputs/teacher_validation_v3.json').write_text(json.dumps(artifact,indent=2))
if __name__=='__main__':
    with WorkerLock():main()
