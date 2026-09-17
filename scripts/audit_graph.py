"""Independently reconcile sampled complete adjacency rows against raw body IDs."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/processed'
def main():
    ids=np.load(D/'body_ids.npy');ip=np.load(D/'indptr.npy',mmap_mode='r');ix=np.load(D/'indices.npy',mmap_mode='r');cw=np.load(D/'counts.npy',mmap_mode='r')
    positions=np.random.default_rng(901).choice(len(ids),256,replace=False);chosen=set(map(int,ids[positions]));allids=set(map(int,ids))
    raw={n:{} for n in chosen};rawfile=ROOT/'data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    reader=pa.ipc.open_file(pa.memory_map(str(rawfile),'r'));n=0
    for k in range(reader.num_record_batches):
        b=reader.get_batch(k);pre=b.column('body_pre').to_numpy();mask=np.isin(pre,list(chosen))
        if not mask.any():continue
        post=b.column('body_post').to_numpy()[mask];weights=b.column('weight').to_numpy()[mask]
        for a,c,w in zip(pre[mask],post,weights):
            if int(c) in allids:raw[int(a)][int(c)]=raw[int(a)].get(int(c),0)+int(w);n+=1
    for pos in positions:
        expected=raw[int(ids[pos])];lo,hi=ip[pos:pos+2]
        imported={int(ids[ix[e]]):int(cw[e]) for e in range(lo,hi)}
        assert imported==expected, 'Direction, identity or weight mismatch for '+str(ids[pos])
    hashes={}
    for name in ['body_ids.npy','indptr.npy','indices.npy','counts.npy','transmitter_sign.npy']:
        h=hashlib.sha256()
        with (D/name).open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        hashes[name]=h.hexdigest()
    result={'pass':True,'method':'256 independently selected presynaptic body IDs; compare every internal outgoing edge and integer weight with raw Feather, using body-ID dictionaries','sampled_neurons':256,'sampled_raw_pairs':n,'total_neurons':len(ids),'total_pairs':len(ix),'total_synapse_weight':int(cw.sum(dtype=np.uint64)),'graph_sha256':hashes,'audited_unix':time.time()}
    (ROOT/'outputs/graph_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
