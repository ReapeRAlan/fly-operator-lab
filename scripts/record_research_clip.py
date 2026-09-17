"""Short labeled neural-actor clip; explicitly excluded from skill acceptance statistics."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from learning_runtime import WorkerLock
from learning_brain import LearningBrain
from learning_env import FlyOperatorEnv
from learning_policy import make_model,decision
from learning_checkpoint import load_checkpoint
from lab_store import atomic_json

def main():
    cfg=json.loads((ROOT/'config/learning.json').read_text());b=LearningBrain();env=FlyOperatorEnv(b,condition='combined',seed=7,stage='switch',split='test')
    model=make_model(env,cfg,7);point=ROOT/'work/learning/checkpoints/integration_v2/current.json';counters=load_checkpoint(point,b,model,env)
    env.train_plasticity=False;x,_=env.reset(options={'mission':'fly_switch_test_2000','record':True});trace=[]
    for i in range(60):
        a,_,_,detail=decision(model,x,env.action_masks(),env.catalog,env.readout.body_ids,True)
        x,r,t,u,info=env.step(a);info.update(decision=detail);trace.append(info)
        if t or u:break
    frames=env.raw.get('capture_directory');env.close()
    atomic_json(ROOT/'outputs/research_clip_trace_v2.json',{'scope':'Short technical sample, not a 30-episode skill evaluation','checkpoint':str(point),'training_steps':counters['steps'],'trace':trace,'frames':frames})
    label=f'Fly Operator | prueba combinada | semilla 7 | checkpoint de prueba | {counters["steps"]} pasos previos | resultado: '+('habilidad completada' if info['success'] else 'muestra limitada a 3 s')
    subprocess.run([sys.executable,str(ROOT/'scripts/export_evaluation_video.py'),'--frames',frames,'--output',str(ROOT/'outputs/FlyOperator_investigacion_v2_1080p30.mp4'),'--label',label],check=True)
if __name__=='__main__':
    with WorkerLock():main()
