from pathlib import Path
p=Path(__file__).resolve().parents[1]/'dashboard/src/main.jsx';s=p.read_text(encoding='utf-8')
marker='function App(){'
component='''function PotentialChart({history}){if(!history?.length)return <p className="padding muted">La serie se registra al consultar esta neurona. Actualiza después de unos pasos para ver su evolución.</p>;const lo=Math.min(-53,...history.map(h=>h.v)),hi=Math.max(-44,...history.map(h=>h.v));return <div className="padding"><h3>Potencial de membrana · mV</h3><svg className="chart" viewBox="0 0 760 240" role="img" aria-label="Evolución del potencial de membrana">{[0,1,2,3,4].map(i=><g key={i}><line x1="65" x2="725" y1={25+i*42} y2={25+i*42} stroke="#e7edf1"/><text x="52" y={29+i*42} textAnchor="end">{fmt(hi-i*(hi-lo)/4,1)}</text></g>)}<polyline fill="none" stroke="#208e89" strokeWidth="2" points={history.map((h,i)=>`${65+i*660/Math.max(1,history.length-1)},${193-(h.v-lo)/(hi-lo)*168}`).join(' ')}/><text x="65" y="222">{fmt(history[0].clock_ms)} ms</text><text x="725" y="222" textAnchor="end">{fmt(history.at(-1).clock_ms)} ms</text></svg></div>}
'''
if 'function PotentialChart' not in s:s=s.replace(marker,component+marker)
old='<Table headers={[\'Origen → destino\',\'Sinapsis\',\'Eficacia ×\',\'Peso simulado\']}'
new='<PotentialChart history={neuron.history}/><div className="padding"><button className="btn" onClick={()=>inspect(neuron.neuron.bodyId)}>Actualizar estado neuronal</button></div>'+old
if '<PotentialChart history=' not in s:
    assert old in s;s=s.replace(old,new)
s=s.replace("['bundle','Paquete de evidencia']","['bundle','Paquete de evidencia'],['video','Clip técnico MP4']")
p.write_text(s,encoding='utf-8')
