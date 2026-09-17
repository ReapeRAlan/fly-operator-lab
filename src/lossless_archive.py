"""NumPy-compatible, atomic NPZ archives with bounded compression work."""
from pathlib import Path
import os,tempfile,time,zipfile
import numpy as np


def atomic_npz(path,arrays,compression_level=1):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as output:
            temporary=Path(output.name)
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=compression_level,allowZip64=True) as archive:
                for name,value in arrays.items():
                    if not name or any(c in name for c in '/\\'):raise ValueError('Invalid array name')
                    with archive.open(name+'.npy','w',force_zip64=True) as member:
                        np.lib.format.write_array(member,np.asanyarray(value),allow_pickle=False)
            output.flush();os.fsync(output.fileno())
        deadline=time.monotonic()+3
        while True:
            try:os.replace(temporary,path);break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
