"""Package source, compiled bridge and evidence, excluding the game and raw data."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def main():
    assert json.loads((OUT/'acceptance.json').read_text())['passed']
    evidence=json.loads((OUT/'provenance.json').read_text())
    evidence['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ['src','scripts','native','config'] for p in (ROOT/sub).glob('*') if p.is_file() and p.suffix in ['.py','.ps1','.java','.cpp','.h','.json']}
    (OUT/'provenance.json').write_text(json.dumps(evidence,indent=2))
    files={ROOT/n for n in ['README.md','requirements.txt','Iniciar Fly Operator.cmd','work/build/flybridge.dll','work/re/symbols.json','work/re/functions.json','work/re/summary.txt','data/processed/audit.json','third_party/json.hpp','third_party/stb_image_write.h'] if (ROOT/n).is_file()}
    for sub in ['src','scripts','native','config','tests','work/decompiled','third_party/minhook-1.3.4']:
        files.update(p for p in (ROOT/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.obj','.pyc','.pdb'])
    files.update((ROOT/'third_party').glob('shiu_*'))
    files.update((ROOT/'third_party').glob('*.provenance.json'))
    files.add(ROOT/'third_party/quantify-neuron-connections.ipynb')
    files.update((ROOT/'data/raw').glob('*.provenance.json'))
    files.update(p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.md','.mp4','.log','.ass'] and p.name!='release_manifest.json')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    target=OUT/'Fly_Operator_prototipo_y_evidencia.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,Path('FlyOperatorLab')/p.relative_to(ROOT))
        z.writestr('FlyOperatorLab/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    release={'path':str(target),'files':len(files)+1,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'contains':'source, DLL and test evidence; game installation and public raw dataset acquired separately'}
    (OUT/'release_manifest.json').write_text(json.dumps(release,indent=2));print(json.dumps(release,indent=2))
if __name__=='__main__':main()
