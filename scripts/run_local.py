"""Use or launch the isolated local game, then run a uniquely named experiment."""
from pathlib import Path
import sys,time,json,subprocess,psutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    candidates=[p for p in psutil.process_iter(['name']) if p.info['name'].lower()=='doorkickers2.exe']
    if candidates:
        session=json.loads((ROOT/'work/session.json').read_text())
        if len(candidates)!=1 or candidates[0].pid!=session['pid'] or abs(candidates[0].create_time()-session['started_unix'])>10:
            raise RuntimeError('Another Door Kickers 2 session is running. Close it before using the isolated laboratory launcher.')
        if hashlib.sha256((ROOT/'work/build/flybridge.dll').read_bytes()).hexdigest()!=session['dll_sha256']:
            raise RuntimeError('The running DLL differs from the built DLL. Relaunch the laboratory game.')
    else:
        from launch import launch
        launch();time.sleep(2)
    args=sys.argv[1:] or ['--episodes','20','--seconds','3']
    name='run_'+time.strftime('%Y%m%d_%H%M%S')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),*args,'--name',name],cwd=ROOT)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
