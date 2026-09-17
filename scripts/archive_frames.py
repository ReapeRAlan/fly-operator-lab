"""Move finished frame captures outside the artifact quota, with a SHA-256 manifest.

The resource guard counts work/frames toward the 8 GB quota. Captures already
exported to video are moved (same volume, no copy) to archive/frames and can be
returned with --restore when a new export needs them.
"""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import atomic_json,file_hash
from learning_runtime import WorkerLock

ACTIVE=ROOT/'work/frames';ARCHIVE=ROOT/'archive/frames';MANIFEST=ARCHIVE/'manifest.json'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    source,target=(ARCHIVE,ACTIVE) if args.restore else (ACTIVE,ARCHIVE)
    with WorkerLock():
        folders=sorted(p for p in source.iterdir() if p.is_dir()) if source.exists() else []
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'version':1,'folders':{}}
        moved=[]
        for folder in folders:
            files=sorted(p for p in folder.rglob('*') if p.is_file())
            entry={'files':{str(p.relative_to(folder)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in files}}
            if args.restore and folder.name in manifest['folders'] and manifest['folders'][folder.name]['files']!=entry['files']:
                raise ValueError(f'Archived capture changed since it was archived: {folder.name}')
            if (target/folder.name).exists():raise FileExistsError(target/folder.name)
            moved.append({'folder':folder.name,'files':len(files),'bytes':sum(x['bytes'] for x in entry['files'].values())})
            if args.dry_run:continue
            target.mkdir(parents=True,exist_ok=True);os.replace(folder,target/folder.name)
            if args.restore:manifest['folders'].pop(folder.name,None)
            else:manifest['folders'][folder.name]={**entry,'archived':time.time()}
        if not args.dry_run and moved:ARCHIVE.mkdir(parents=True,exist_ok=True);atomic_json(MANIFEST,manifest)
    print(json.dumps({'restore':args.restore,'dry_run':args.dry_run,'folders':len(moved),'files':sum(m['files'] for m in moved),
                      'gib':round(sum(m['bytes'] for m in moved)/2**30,3)},indent=2))

if __name__=='__main__':main()
