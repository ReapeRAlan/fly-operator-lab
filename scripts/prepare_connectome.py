"""Preserve all published within-neuron edges for all 166700 classified CNS neurons.

The raw graph also contains unclassified segments and glia. Their boundary edges
are audited separately; absence of a superclass does not prove a segment is glia.
No region, cell type, edge weight or synapse-count pruning is performed.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; OUT=ROOT/'data/processed'

def main():
    started=time.perf_counter()
    annotations=feather.read_table(RAW/'body-annotations-male-cns-v1.0-minconf-0.5.feather').to_pandas()
    cells=annotations.loc[annotations.superclass.notna()].sort_values('bodyId').reset_index(drop=True)
    assert len(cells)==166700, 'Version 1.0 inventory changed; review source before proceeding'
    assert not cells.bodyId.duplicated().any()
    ids=cells.bodyId.to_numpy(np.int64); np.save(OUT/'body_ids.npy',ids)
    feather.write_feather(pa.Table.from_pandas(cells,preserve_index=False),OUT/'neurons.feather')
    nts=feather.read_table(RAW/'body-neurotransmitters-male-cns-v1.0.feather').to_pandas().set_index('body')
    nt=nts.reindex(ids); labels=nt.consensus_nt.fillna(nt.ground_truth).fillna(nt.predicted_nt).fillna('unknown')
    signs=np.where(labels.isin(['gaba','glutamate','histamine']),-1.,1.).astype(np.float32)
    np.save(OUT/'transmitter_sign.npy',signs)
    reader=pa.ipc.open_file(pa.memory_map(str(RAW/'connectome-weights-male-cns-v1.0-minconf-0.5.feather'),'r'))
    stats={'raw_rows':0,'internal_rows':0,'internal_synapses':0,'boundary_rows':0,'boundary_synapses':0,'unclassified_rows':0,'unclassified_synapses':0,'nonpositive_rows':0}
    handles=[(OUT/(x+'.bin')).open('wb') for x in ['pre_index','post_index','synapse_count']]
    try:
        for k in range(reader.num_record_batches):
            batch=reader.get_batch(k)
            pre=batch.column('body_pre').to_numpy(); post=batch.column('body_post').to_numpy(); w=batch.column('weight').to_numpy()
            pi=np.searchsorted(ids,pre); qi=np.searchsorted(ids,post)
            pv=(pi<len(ids)) & (ids[np.minimum(pi,len(ids)-1)]==pre)
            qv=(qi<len(ids)) & (ids[np.minimum(qi,len(ids)-1)]==post)
            good=pv&qv; boundary=pv^qv; rest=~(pv|qv)
            stats['raw_rows']+=len(w); stats['nonpositive_rows']+=int(np.sum(w<=0))
            for name,mask in [('internal',good),('boundary',boundary),('unclassified',rest)]:
                stats[name+'_rows']+=int(mask.sum()); stats[name+'_synapses']+=int(w[mask].sum(dtype=np.int64))
            assert np.all(w[good]<=np.iinfo(np.uint32).max)
            pi[good].astype(np.uint32).tofile(handles[0]); qi[good].astype(np.uint32).tofile(handles[1]); w[good].astype(np.uint32).tofile(handles[2])
            if k%200==0: print('BATCH',k,'/',reader.num_record_batches,stats,flush=True)
    finally:
        for h in handles: h.close()
    assert stats['nonpositive_rows']==0
    pre=np.memmap(OUT/'pre_index.bin',dtype=np.uint32,mode='r')
    post=np.memmap(OUT/'post_index.bin',dtype=np.uint32,mode='r')
    weight=np.memmap(OUT/'synapse_count.bin',dtype=np.uint32,mode='r')
    graph=csr_matrix((weight.astype(np.int64),(pre,post)),shape=(len(ids),len(ids)))
    graph.sum_duplicates(); graph.sort_indices()
    assert int(graph.sum(dtype=np.int64))==stats['internal_synapses']
    # Retain integer counts separately from dynamical weights.
    np.save(OUT/'indptr.npy',graph.indptr.astype(np.int64)); np.save(OUT/'indices.npy',graph.indices.astype(np.int32)); np.save(OUT/'counts.npy',graph.data.astype(np.uint32))
    stats.update({'neurons':len(ids),'csr_edges':int(graph.nnz),'duplicate_pairs_aggregated':stats['internal_rows']-graph.nnz,'annotation_records':len(annotations),'glia_records':int((annotations.status=='Glia').sum()),'classified_status_counts':cells.status.fillna('none').value_counts().to_dict(),'superclass_counts':cells.superclass.value_counts().to_dict(),'nt_labels':labels.value_counts().to_dict(),'unknown_nt_count':int((labels=='unknown').sum()),'excluded_annotation_status_counts':annotations.loc[annotations.superclass.isna(),'status'].fillna('none').value_counts().to_dict(),'boundary_policy':'All 166700 rows with non-null superclass are simulated, including provisional tbc classes. All pairs between these nodes are retained without an extra weight threshold. Unclassified endpoints are outside this identified-neuron model, including 516 Traced rows without superclass. The full raw graph is retained.','source_threshold':'Published minconf-0.5 dataset; no additional confidence filter.','weights_sha256':hashlib.sha256(graph.data.tobytes()).hexdigest(),'seconds':time.perf_counter()-started})
    (OUT/'audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8'); print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__': main()
