import React,{useEffect,useMemo,useState} from 'react';
import {FAMILY_COLORS,FAMILY_SHORT,SIGN_COLORS,ACCENT,INK,BLUE_RAMP,fmt,compact,pct,useVisiblePoll,useTip,TipRow,Sparkline,Legend} from './viz.jsx';
import {withRun} from './Provenance.jsx';

const api=async path=>{const r=await fetch('/api/'+path);if(!r.ok)throw Error(await r.text());return r.json()};
const TOTAL_NEURONS=166700;

function Kpis({data}){
 const series=data?.activity_series||[],last=series.at(-1)||{},groups=data?.groups||[];
 const validation=groups.filter(g=>g.split==='validation'),n=validation.reduce((a,g)=>a+g.n,0),ok=validation.reduce((a,g)=>a+(g.successes||0),0);
 const tiles=[
  ['Impulsos en 50 ms',compact(last.spikes),series.map(s=>s.spikes),'de toda la red simulada'],
  ['Neuronas activas',pct(last.active/TOTAL_NEURONS),series.map(s=>s.active),`${fmt(last.active)} de ${fmt(TOTAL_NEURONS)}`],
  ['Cálculo neuronal',last.wall==null?'—':`${fmt(last.wall,2)} s`,series.map(s=>s.wall),'por paso de 50 ms de juego'],
  ['Éxito en validación',n?pct(ok/n,0):'—',[],n?`${ok} de ${n} episodios reservados`:'sin validaciones todavía']];
 return <div className="kpi-row">{tiles.map(([label,value,values,note])=><div className="kpi" key={label}><span>{label}</span><strong>{value}</strong>{values.length>1?<Sparkline values={values}/>:<div className="spark-space"/>}<small>{note}</small></div>)}</div>}

// Five stages: game inputs -> gateway families -> network families -> descending neurons -> actions.
function Flow({structure,live}){
 const [tip,show,hide]=useTip();
 const families=structure.families,index=Object.fromEntries(families.map((f,i)=>[f.key,i]));
 const syn=(a,b)=>structure.synapses_excitatory[index[a]][index[b]]+structure.synapses_inhibitory[index[a]][index[b]];
 const liveFamily=Object.fromEntries((live?.families||[]).map(f=>[f.key,f]));
 const activeShare=key=>{const f=liveFamily[key];return f?f.active/f.neurons:null};
 const gatewayKeys=['visual_projection','sensory'],networkKeys=['optic','central','vnc','ascending','other'];
 const decision=live?.decision||{},labels=live?.action_labels||[];
 const actions=(decision.probabilities||[]).map((p,i)=>({i,p,label:labels[i]||String(i)})).filter(a=>decision.mask?.[a.i]).sort((a,b)=>b.p-a.p).slice(0,5);
 const W=1000,H=440,colX=[30,230,440,640,810],nodeW=14,top=46,bottom=H-24;
 // Each node reserves at least `slot` px so its two-line label never collides with the next node.
 const place=(items,sizes,gap=10,slot=0)=>{const steps=sizes.map(h=>Math.max(h,slot));const total=steps.reduce((a,b)=>a+b,0)+gap*(items.length-1);let y=top+Math.max(0,(bottom-top-total)/2);return items.map((item,k)=>{const node={...item,y,h:sizes[k]};y+=steps[k]+gap;return node})};
 const drive=live?.input_drive_hz||{},maxDrive=Math.max(1,...structure.input_groups.map(g=>drive[g.key]||0));
 const inputs=place(structure.input_groups.map(g=>({...g,col:0,value:drive[g.key]||0})),structure.input_groups.map(g=>6+34*((drive[g.key]||0)/maxDrive)),8,30);
 const familyNode=(key,col)=>({key,col,label:families[index[key]].label,neurons:families[index[key]].neurons});
 const gateways=place(gatewayKeys.map(k=>familyNode(k,1)),[120,120],40),network=place(networkKeys.map(k=>familyNode(k,2)),networkKeys.map(()=>52),14);
 const dn=place([familyNode('descending',3)],[150]),acts=place(actions.map(a=>({...a,key:'action'+a.i,col:4})),actions.map(a=>8+46*a.p),10,30);
 const byKey=Object.fromEntries([...gateways,...network,...dn].map(n=>[n.key,n]));
 const links=[];
 for(const node of inputs){const ports=structure.gateways[node.key]||{},total=Object.values(ports).reduce((a,b)=>a+b,0)||1;
  for(const [family,count] of Object.entries(ports)){if(!byKey[family])continue;links.push({from:node,to:byKey[family],stage:0,value:node.value*count/total,label:`${node.label} → ${byKey[family].label}`,unit:'Hz de estímulo sobre la línea base'})}}
 for(const g of gateways){for(const n of network)links.push({from:g,to:n,stage:1,value:syn(g.key,n.key),pre:g.key,label:`${g.label} → ${n.label}`,unit:'sinapsis anatómicas'});
  links.push({from:g,to:dn[0],stage:2,value:syn(g.key,'descending'),pre:g.key,label:`${g.label} → Descendentes`,unit:'sinapsis anatómicas'})}
 for(const n of network)links.push({from:n,to:dn[0],stage:2,value:syn(n.key,'descending'),pre:n.key,label:`${n.label} → Descendentes`,unit:'sinapsis anatómicas'});
 for(const a of acts)links.push({from:dn[0],to:a,stage:3,value:a.p,label:`Descendentes → ${a.label}`,unit:'probabilidad del actor',probability:true});
 const maxByStage=[0,1,2,3].map(s=>Math.max(1e-9,...links.filter(l=>l.stage===s).map(l=>l.value)));
 const widthOf=l=>l.stage===0||l.stage===3?1+11*l.value/maxByStage[l.stage]:1+11*Math.sqrt(l.value/maxByStage[l.stage]);
 const maxActive=Math.max(1e-9,...families.map(f=>activeShare(f.key)||0));
 const opacityOf=l=>l.pre&&activeShare(l.pre)!=null?.18+.55*activeShare(l.pre)/maxActive:.45;
 // Stack link ends along each node so ribbons do not pile on one point.
 const offsets=new Map();const anchor=(node,side,w)=>{const key=node.key+side,used=offsets.get(key)||0;offsets.set(key,used+w);return node.y+Math.min(node.h,used+w/2+2)};
 const drawn=links.filter(l=>l.value>0).sort((a,b)=>a.stage-b.stage).map(l=>{const w=widthOf(l),x0=colX[l.from.col]+nodeW,x1=colX[l.to.col];const y0=anchor(l.from,'out',w),y1=anchor(l.to,'in',w),mx=(x0+x1)/2;return {...l,w,d:`M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1}`}});
 const color=node=>FAMILY_COLORS[node.key]||INK.muted;
 const nodeTip=(e,node,kind)=>{if(kind==='input')show(e,<><TipRow value={`${fmt(node.value)} Hz`} label={`Estímulo · ${node.label}`}/><div className="tip-meta">Suma sobre sus puertas neuronales, por encima de la línea base</div></>);
  else if(kind==='action')show(e,<><TipRow value={pct(node.p)} label={node.label}/><div className="tip-meta">{decision.executed===node.i?'Orden ejecutada en este paso':'Probabilidad del actor lineal'}</div></>);
  else{const f=liveFamily[node.key];show(e,<><TipRow color={color(node)} value={f?pct(f.active/f.neurons):'—'} label={`${node.label} activas`}/><TipRow value={f?compact(f.spikes):'—'} label="impulsos en 50 ms"/><div className="tip-meta">{fmt(node.neurons)} neuronas</div></>)}};
 const stageNames=['Entradas del juego','Compuertas','Red interna','Salida motora','Acción'];
 return <div className="viz-wrap flow-scroll"><svg className="flow" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Flujo desde las entradas del juego hasta las acciones">
  {stageNames.map((name,i)=><text key={name} className="stage-title" x={colX[i]+(i===4?0:nodeW/2)} y="20" textAnchor={i===0?'start':i===4?'start':'middle'}>{name}</text>)}
  {drawn.map((l,k)=><path key={k} d={l.d} fill="none" stroke={l.stage===0?color(l.to):l.stage===3?ACCENT:color(l.from)} strokeWidth={l.w} strokeOpacity={opacityOf(l)} onPointerMove={e=>show(e,<><TipRow value={l.probability?pct(l.value):compact(l.value)} label={l.label}/><div className="tip-meta">{l.unit}{l.pre&&activeShare(l.pre)!=null?` · origen ${pct(activeShare(l.pre))} activo`:''}</div></>)} onPointerLeave={hide}/>)}
  {inputs.map(n=><g key={n.key} onPointerMove={e=>nodeTip(e,n,'input')} onPointerLeave={hide}><rect x={colX[0]} y={n.y} width={nodeW} height={n.h} rx="3" fill={n.value?INK.secondary:INK.quiet}/><text x={colX[0]+nodeW+8} y={n.y+Math.max(n.h,24)/2}>{n.label}</text><text className="value" x={colX[0]+nodeW+8} y={n.y+Math.max(n.h,24)/2+13}>{n.value?compact(n.value)+' Hz':'sin señal'}</text></g>)}
  {[...gateways,...network,...dn].map(n=>{const col=n.col,f=liveFamily[n.key];return <g key={n.key} onPointerMove={e=>nodeTip(e,n,'family')} onPointerLeave={hide}><rect x={colX[col]} y={n.y} width={nodeW} height={n.h} rx="3" fill={color(n)}/><rect x={colX[col]-3} y={n.y-3} width={nodeW+6} height={n.h+6} fill="transparent"/><text x={colX[col]+nodeW+8} y={n.y+14}>{FAMILY_SHORT[n.key]}</text><text className="value" x={colX[col]+nodeW+8} y={n.y+28}>{f?pct(f.active/f.neurons)+' activas':'—'}</text></g>})}
  {acts.map(a=><g key={a.i} onPointerMove={e=>nodeTip(e,a,'action')} onPointerLeave={hide}><rect x={colX[4]} y={a.y} width={nodeW} height={a.h} rx="3" fill={decision.executed===a.i?ACCENT:INK.quiet}/><text x={colX[4]+nodeW+8} y={a.y+Math.max(a.h,24)/2}>{actions.length===1&&a.label==='wait'?'esperar (orden en curso)':a.label.replace('move · ','mover ')}</text><text className="value" x={colX[4]+nodeW+8} y={a.y+Math.max(a.h,24)/2+13}>{pct(a.p)}{decision.executed===a.i?' · ejecutada':''}</text></g>)}
  {!acts.length&&<text x={colX[4]} y={H/2}>Sin decisión libre</text>}
 </svg>{tip}</div>}

function Matrix({structure}){
 const [mode,setMode]=useState('synapses');const [tip,show,hide]=useTip();
 const families=structure.families,F=families.length,cell=40,left=92,top=76;
 const total=(i,j)=>structure.synapses_excitatory[i][j]+structure.synapses_inhibitory[i][j];
 const values=families.flatMap((_,i)=>families.map((_,j)=>total(i,j))),max=Math.max(...values),min=Math.min(...values.filter(v=>v>0));
 // Log scale spans the observed range so the seven steps separate real differences.
 const logColor=v=>{if(!v)return '#f4f7f8';const t=(Math.log10(v)-Math.log10(min))/Math.max(1e-9,Math.log10(max)-Math.log10(min));return BLUE_RAMP[Math.max(0,Math.min(BLUE_RAMP.length-1,Math.floor(t*BLUE_RAMP.length)))]};
 const diverging=f=>{const blue=['#256abf','#6da7ec','#b7d3f6'],red=['#f4c2c1','#e97d7c','#c93a39'];if(f<.4)return blue[Math.min(2,Math.floor(f/.4*3))];if(f<=.6)return '#eceae5';return red[Math.min(2,Math.floor((f-.6)/.4*3))]};
 const fill=(i,j)=>{const t=total(i,j);return mode==='synapses'?logColor(t):t?diverging(structure.synapses_inhibitory[i][j]/t):'#f4f7f8'};
 const W=left+F*cell+10,H=top+F*cell+10;
 return <><div className="seg" role="group" aria-label="Medida de la matriz"><button className={mode==='synapses'?'on':''} onClick={()=>setMode('synapses')}>Sinapsis</button><button className={mode==='balance'?'on':''} onClick={()=>setMode('balance')}>Balance E/I</button></div>
 <div className="viz-wrap matrix-scroll"><svg className="matrix" viewBox={`0 0 ${W} ${H}`} style={{maxWidth:W}} role="img" aria-label="Matriz de conectividad entre familias neuronales">
  <text className="axis-title" x={left} y="14">Postsináptica →</text><text className="axis-title" x="4" y={top-8}>Presináptica ↓</text>
  {families.map((f,j)=><g key={'c'+f.key} transform={`translate(${left+j*cell+cell/2},${top-10}) rotate(-45)`}><text>{FAMILY_SHORT[f.key]}</text></g>)}
  {families.map((f,i)=><g key={'r'+f.key}><circle cx="8" cy={top+i*cell+cell/2} r="4" fill={FAMILY_COLORS[f.key]}/><text x="17" y={top+i*cell+cell/2+4}>{FAMILY_SHORT[f.key]}</text>
   {families.map((g,j)=>{const t=total(i,j),inh=structure.synapses_inhibitory[i][j];return <rect key={g.key} x={left+j*cell+1} y={top+i*cell+1} width={cell-2} height={cell-2} rx="3" fill={fill(i,j)}
    onPointerMove={e=>show(e,<><TipRow value={compact(t)} label={`sinapsis ${f.label} → ${g.label}`}/><TipRow color={SIGN_COLORS.excitatory} line value={pct(t?1-inh/t:null)} label="excitatorias"/><TipRow color={SIGN_COLORS.inhibitory} line value={pct(t?inh/t:null)} label="inhibitorias"/><div className="tip-meta">{fmt(structure.connections[i][j])} pares de neuronas conectados</div></>)} onPointerLeave={hide}/>})}</g>)}
 </svg>{tip}</div>
 {mode==='synapses'?<div className="scale-legend"><span>Menos</span>{BLUE_RAMP.map(c=><i key={c} style={{background:c}}/>)}<span>Más sinapsis (escala logarítmica de {compact(min)} a {compact(max)})</span></div>
  :<div className="scale-legend"><span>Mayoría excitatoria</span>{['#256abf','#6da7ec','#b7d3f6','#eceae5','#f4c2c1','#e97d7c','#c93a39'].map(c=><i key={c} style={{background:c}}/>)}<span>Mayoría inhibitoria</span></div>}
 <details className="raw"><summary>Ver la matriz como tabla</summary><div className="table-scroll"><table><thead><tr><th>Pre \ Post</th>{families.map(f=><th key={f.key}>{FAMILY_SHORT[f.key]}</th>)}</tr></thead><tbody>{families.map((f,i)=><tr key={f.key}><td>{f.label}</td>{families.map((g,j)=><td key={g.key}>{compact(total(i,j))}</td>)}</tr>)}</tbody></table></div></details></>}

function FamilyActivity({structure,live}){
 const [tip,show,hide]=useTip();
 const rows=(live?.families||[]).map(f=>({...f,label:structure.families.find(x=>x.key===f.key)?.label,share:f.active/f.neurons}));
 if(!rows.length)return <div className="empty">Esperando una ventana neuronal coherente</div>;
 const max=Math.max(...rows.map(r=>r.share),1e-9);
 return <div className="viz-wrap family-bars">{rows.map(r=><div className="fbar" key={r.key} tabIndex="0" onPointerMove={e=>show(e,<><TipRow value={pct(r.share)} label={`${r.label} activas`}/><TipRow value={compact(r.spikes)} label="impulsos en 50 ms"/><div className="tip-meta">{fmt(r.active)} de {fmt(r.neurons)} neuronas</div></>)} onPointerLeave={hide}>
  <label><i style={{background:FAMILY_COLORS[r.key]}}/>{r.label}</label><div className="track"><b style={{width:`${Math.max(1.5,100*r.share/max)}%`}}/></div><span>{pct(r.share)}</span></div>)}{tip}</div>}

export default function Circuit({data,run}){
 const [structure,setStructure]=useState(null),[live,setLive]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api('circuit/structure').then(setStructure).catch(e=>setError('No se pudo cargar la estructura: '+e.message))},[]);
 useVisiblePoll(async alive=>{const next=await api(withRun('circuit/live',run));if(alive())setLive(next)},2500,[run]);
 const s=data?.status||{},liveNow=live?.context?.freshness==='live'&&live?.coherent;
 return <div className="circuit-view">
  {error&&<div className="notice" role="alert">{error}</div>}
  <Kpis data={data}/>
  <article className="panel"><div className="panel-heading"><div><h2>{liveNow?'Del juego a la acción, en vivo':'Circuito anatómico y última decisión registrada'}</h2><p>{liveNow?(s.stage?`Etapa ${s.stage} · ${s.condition||'—'} · semilla ${s.seed??'—'}`:'Esperando el entrenador'):'La anatomía se conserva; no se presentan tasas, gains ni probabilidades como actividad actual.'}</p></div><span className="tag">{liveNow?'Actualiza cada 2.5 s':'Solo lectura'}</span></div>
   {structure&&liveNow?<Flow structure={structure} live={live}/>:<div className="circuit-archive"><strong>{live?.action_label?<>Última decisión: {live.action_label}</>:'Actividad neuronal no disponible'}</strong><span>La anatomía y la última decisión durable siguen disponibles; no se dibuja telemetría como actual.</span></div>}
   <div className="flow-notes"><span><b>La matriz:</b> conserva la anatomía agrupada del conectoma.</span><span className="muted">Actividad, gains y probabilidades sólo aparecen para un snapshot actual y coherente.</span></div>
  </article>
  <div className="circuit-grid">
   <article className="panel"><div className="panel-heading"><div><h2>Actividad por familia</h2><p>{liveNow?'Fracción de neuronas que dispararon en los últimos 50 ms':'No se reconstruye actividad para campañas archivadas u obsoletas.'}</p></div></div>{structure&&(liveNow?<FamilyActivity structure={structure} live={live}/>:<div className="empty">Actividad neuronal no disponible para este contexto.</div>)}</article>
   <article className="panel"><div className="panel-heading"><div><h2>Matriz de conectividad</h2><p>25,582,938 conexiones agrupadas en 8 familias · signo según neurotransmisor predicho</p></div></div>{structure?<Matrix structure={structure}/>:<div className="empty">Cargando…</div>}</article>
  </div>
 </div>}
