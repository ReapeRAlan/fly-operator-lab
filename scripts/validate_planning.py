"""Verify real time/movement and client-frame synchronization, without neural training."""
from pathlib import Path
import sys,json,math,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bridge_client import Bridge
from lab_store import atomic_json

with Bridge() as bridge:
    initial=bridge.new_mission('fly_move_train_7',record=True);trace=[]
    for step in range(25):
        observation=bridge.step({'action':'move','destination':[-2,0,0]} if step==0 else {'action':'wait'},50)
        trace.append({key:observation.get(key) for key in ('sim_time_ms','game_flags','client_view','operator','frame','capture_directory')})
        assert observation['frame']['client_sim_time_ms']==observation['sim_time_ms']
        assert observation['sim_time_ms']==initial['sim_time_ms']+(step+1)*50
    last=trace[-1];distance=math.dist(initial['operator']['position'],last['operator']['position'])
    assert distance>.5
    before=bridge.request('observe');time.sleep(.4);after=bridge.request('observe')
    assert before['sim_time_ms']==after['sim_time_ms']
    assert before['operator']['position']==after['operator']['position']
    atomic_json(ROOT/'outputs/planning_validation.json',{'session':bridge.session,'passed':True,'movement_m':distance,'trace':trace,'pause_keeps_time_and_position':True,'frame':str(Path(last['capture_directory'])/'frame_000024.png')})
    print(json.dumps({'passed':True,'distance_m':distance,'server_ms':last['sim_time_ms'],'frame':last['frame'],'path':str(Path(last['capture_directory'])/'frame_000024.png')}))
