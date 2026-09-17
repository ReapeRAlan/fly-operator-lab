"""Low-overhead local samples; does not change workloads or power settings."""
from pathlib import Path
import sys,time,json,argparse,subprocess,statistics
import psutil
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}
def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=30);p.add_argument('--name',required=True);a=p.parse_args()
    processes={};psutil.cpu_percent();rows=[];gpu={}
    state=read(ROOT/'work/learning/status.json');game=read(ROOT/'work/session.json')
    for pid in (state.get('pid'),game.get('pid')):
        try:
            proc=psutil.Process(pid);proc.cpu_percent();processes[pid]=(proc,proc.name(),proc.nice(),proc.cpu_affinity())
        except (psutil.Error,TypeError):pass
    for i in range(a.seconds):
        time.sleep(1);cpu=psutil.cpu_percent();memory=psutil.virtual_memory();status=read(ROOT/'work/learning/status.json');latest=read(ROOT/'work/learning/latest.json')
        top=[];lab=[]
        for proc,name,priority,affinity in processes.values():
            try:
                value={'pid':proc.pid,'name':name,'cpu_machine_pct':proc.cpu_percent()/psutil.cpu_count(),'rss_gb':proc.memory_info().rss/1e9}
                top.append(value)
                if proc.pid==status.get('pid') or value['name']=='DoorKickers2.exe':
                    value.update(priority=priority,affinity=affinity);lab.append(value)
            except psutil.Error:pass
        if i%5==0:
            try:
                out=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,creationflags=subprocess.CREATE_NO_WINDOW)
                values=[float(x.strip()) for x in out.stdout.splitlines()[0].split(',')]
                gpu=dict(zip(['temperature_c','utilization_pct','memory_mib','power_w'],values))
            except Exception as exc:gpu={'error':type(exc).__name__}
        rows.append({'time':time.time(),'cpu_pct':cpu,'available_ram_gb':memory.available/1e9,'committed_ram_pct':memory.percent,'gpu':gpu,
          'top_cpu':sorted(top,key=lambda r:r['cpu_machine_pct'],reverse=True)[:5],'lab':lab,
          'state':status.get('state'),'phase':status.get('phase'),'episode':latest.get('episode'),'sequence':latest.get('sequence'),
          'steps':status.get('steps'),'step_wall_s':status.get('neural_wall_seconds')})
        if i%10==0:print(json.dumps({'sample':i+1,'cpu_pct':cpu,'ram_gb':memory.available/1e9,'steps':status.get('steps')}),flush=True)
    summary={'samples':len(rows),'cpu_mean_pct':statistics.mean(r['cpu_pct'] for r in rows),'cpu_max_pct':max(r['cpu_pct'] for r in rows),
      'ram_available_min_gb':min(r['available_ram_gb'] for r in rows),'gpu_max_c':max((r['gpu'].get('temperature_c',0) for r in rows),default=0),
      'gpu_mean_pct':statistics.mean(r['gpu'].get('utilization_pct',0) for r in rows),
      'start_steps':rows[0]['steps'],'end_steps':rows[-1]['steps'],'states':sorted({str(r['state']) for r in rows})}
    (ROOT/'outputs'/f'performance_{a.name}.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
