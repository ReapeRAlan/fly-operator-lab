"""Append real episodes until the video contains exactly 60,000 simulated ms."""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    folders=[ROOT/'outputs/recorded_validation']
    for attempt in range(10):
        rows=[]
        for folder in folders:
            summary=json.loads((folder/'summary.json').read_text())
            if not summary['passed']:raise RuntimeError('Source run did not finish successfully')
            rows.extend(json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines())
        missing=1800-len(rows);remaining_ms=60000-sum(r['activity']['simulated_ms'] for r in rows)
        if missing==0:
            assert remaining_ms==0
            break
        assert missing>0
        last_dt=int(remaining_ms-sum([33,33,34][i%3] for i in range(missing-1)))
        assert 1<=last_dt<=50
        name=f'recorded_tail_{attempt+1:02}'
        print('CAPTURE TAIL',name,'frames',missing,'last_dt',last_dt,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--episodes','1','--record','--frames',str(missing),'--last-dt',str(last_dt),'--name',name],cwd=ROOT,check=True)
        folders.append(ROOT/'outputs'/name)
    else:raise RuntimeError('Could not complete the one-minute recording in ten additional episodes')
    subprocess.run([sys.executable,str(ROOT/'scripts/export_video.py'),*[str(x) for x in folders],'--frames','1800'],cwd=ROOT,check=True)
if __name__=='__main__':main()
