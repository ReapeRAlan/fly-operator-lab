"""Compare lossless compression on real saved arrays without starting another brain."""
from pathlib import Path
import sys,json,time,os,hashlib,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,psutil
from lossless_archive import atomic_npz
from lab_store import atomic_json

proc=psutil.Process();proc.nice(psutil.IDLE_PRIORITY_CLASS);proc.cpu_affinity(proc.cpu_affinity()[-1:])
folder=ROOT/'work/checkpoint_benchmark';folder.mkdir(exist_ok=True)
manifest_path=ROOT/'work/learning/checkpoints/common_43/current.json';manifest=json.loads(manifest_path.read_text())
sources=[manifest_path.parent/str(manifest['generation'])/'brain.npz',ROOT/'work/learning/examples_19.npz'];results=[]
for source in sources:
    with np.load(source,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    row={'source':str(source),'arrays':len(arrays),'raw_bytes':sum(a.nbytes for a in arrays.values())}
    durations={6:[],1:[]}
    for repeat in range(3):
        for level in ((6,1) if repeat%2==0 else (1,6)):
            target=folder/(source.stem+f'_level{level}.npz');start=time.perf_counter()
            if level==6:
                with target.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            else:atomic_npz(target,arrays,compression_level=1)
            durations[level].append(time.perf_counter()-start)
            with np.load(target,allow_pickle=False) as z:
                for key,value in arrays.items():
                    restored=z[key]
                    assert restored.dtype==value.dtype and restored.shape==value.shape and restored.tobytes()==value.tobytes(),key
            row[f'level_{level}']={'seconds':statistics.median(durations[level]),'samples':durations[level], 'bytes':target.stat().st_size,'all_arrays_bitwise_equal':True}
    row['compression_speedup']=row['level_6']['seconds']/row['level_1']['seconds'];results.append(row);print(json.dumps(row),flush=True)
atomic_json(ROOT/'outputs/checkpoint_io_benchmark.json',{'scope':'Three sequential alternating-order samples per dataset, medians, CPU affinity 1, IDLE; compression time only, not end-to-end neural speed','results':results,'duplicate_example_write_removed':True})
