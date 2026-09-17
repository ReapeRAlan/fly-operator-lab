"""Create a reproducible, openly artificial input/output mapping over real neurons."""
from pathlib import Path
import sys,json,hashlib
import pyarrow.feather as feather
ROOT=Path(__file__).resolve().parents[1]
def main():
    df=feather.read_table(ROOT/'data/processed/neurons.feather').to_pandas()
    used=set(); inputs={}
    side=df.rootSide.fillna(df.somaSide)
    specs=[('vision_left',(df.superclass=='ol_sensory')&(side=='L')),
           ('vision_right',(df.superclass=='ol_sensory')&(side=='R')),
           ('proprioception',(df.superclass=='vnc_sensory')&(df['class']=='mechanosensory_proprioceptive')),
           ('damage',(df.superclass=='cb_sensory')&(df['class'].fillna('').str.contains('mechanosensory'))),
           ('ammo',df.superclass=='cb_sensory'),('tonic',df.superclass=='vnc_sensory')]
    for name,mask in specs:
        selected=[int(i) for i in df.index[mask] if int(i) not in used][:128]
        if not selected: raise ValueError('No inputs for '+name)
        used.update(selected); inputs[name]={'indices':selected,'body_ids':[int(df.loc[i,'bodyId']) for i in selected]}
    # Anatomical descending output neurons assigned to channels by stable hash.
    names=['move','stop','turn_left','turn_right','aim','fire','reload']; outputs={k:{'indices':[],'body_ids':[]} for k in names}
    for i in df.index[df.superclass=='descending_neuron']:
        body=int(df.loc[i,'bodyId']); channel=int.from_bytes(hashlib.sha256(str(body).encode()).digest()[:4],'little')%len(names)
        outputs[names[channel]]['indices'].append(int(i)); outputs[names[channel]]['body_ids'].append(body)
    conf={'version':1,'meaning':'Engineered fixed semantic adapter, not biological evidence of weapon concepts. All 166700 neurons are simulated; only stimulation/readout ports are selected. No direct sensor-to-action rule and no learned policy.','inputs':inputs,'outputs':outputs,'selection':'First 128 available matching sensory annotations, sorted bodyId; descending output groups use SHA256(bodyId) mod 7.','stimulus_hz':{'baseline':5,'gain':145},'silent_output_action':'stop'}
    (ROOT/'config/adapter.json').write_text(json.dumps(conf,indent=2),encoding='utf-8')
    print({k:len(v['indices']) for k,v in inputs.items()}); print({k:len(v['indices']) for k,v in outputs.items()})
if __name__=='__main__': main()
