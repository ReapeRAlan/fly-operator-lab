"""Move finished campaigns out of the live SQLite into compressed JSONL, and reclaim the space.

Evidence is preserved: every episode and step row is written to archive/<run>/{episodes,steps}.jsonl.gz
with its row id, verified by re-reading the archive, and only then deleted from the database.
"""
from pathlib import Path
import sys,json,gzip,sqlite3,argparse,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import RUNTIME

def export(db,table,run,path):
    rows=0;digest=hashlib.sha256()
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as out:
        for ident,payload in db.execute(f'SELECT id,payload FROM {table} WHERE run=? ORDER BY id',(run,)):
            line=json.dumps({'id':ident,'payload':json.loads(payload)},allow_nan=False)
            out.write(line+'\n');digest.update(line.encode());rows+=1
    return rows,digest.hexdigest()

def verify(path,rows,digest):
    seen=0;check=hashlib.sha256()
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            check.update(line.rstrip('\n').encode());seen+=1
    return seen==rows and check.hexdigest()==digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--keep',nargs='*',default=[],help='Runs to keep in the live database')
    parser.add_argument('--drop',nargs='*',default=[],help='Runs to delete without archiving (smoke tests)')
    parser.add_argument('--destination',default='archive/experiments')
    parser.add_argument('--apply',action='store_true',help='Without it, only report what would move')
    args=parser.parse_args()
    database=RUNTIME/'experiments.sqlite'
    db=sqlite3.connect(database);db.execute('PRAGMA journal_mode=WAL')
    runs=[r[0] for r in db.execute('SELECT DISTINCT run FROM steps UNION SELECT DISTINCT run FROM episodes')]
    moving=[r for r in runs if r not in args.keep and r not in args.drop]
    report={'database':str(database),'size_before_gb':round(database.stat().st_size/1e9,3),'keep':args.keep,'drop':args.drop,'archived':{},'dropped':{}}
    print(json.dumps({'runs':runs,'moving':moving,'dropping':args.drop},indent=2))
    if not args.apply:
        print('Dry run: pass --apply to archive and delete.');return 0
    root=ROOT/args.destination;root.mkdir(parents=True,exist_ok=True)
    for run in moving:
        folder=root/run.replace('/','_');folder.mkdir(parents=True,exist_ok=True)
        entry={}
        for table in ('episodes','steps'):
            path=folder/f'{table}.jsonl.gz'
            rows,digest=export(db,table,run,path)
            if not verify(path,rows,digest):raise RuntimeError(f'Archive verification failed for {run}/{table}')
            entry[table]={'rows':rows,'sha256':digest,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size}
        (folder/'manifest.json').write_text(json.dumps({'run':run,'archived_at':time.time(),**entry},indent=2))
        for table in ('episodes','steps'):db.execute(f'DELETE FROM {table} WHERE run=?',(run,))
        db.commit();report['archived'][run]=entry
    for run in args.drop:
        counts={table:db.execute(f'SELECT count(*) FROM {table} WHERE run=?',(run,)).fetchone()[0] for table in ('episodes','steps')}
        for table in ('episodes','steps'):db.execute(f'DELETE FROM {table} WHERE run=?',(run,))
        db.commit();report['dropped'][run]=counts
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.execute('VACUUM');db.commit();db.close()
    report['size_after_gb']=round(database.stat().st_size/1e9,3)
    out=ROOT/'outputs/archive_runs.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('size_before_gb','size_after_gb','dropped')},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
