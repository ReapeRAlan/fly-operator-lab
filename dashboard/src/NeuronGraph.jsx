import React,{useState} from 'react';
import {FAMILY_COLORS,FAMILY_SHORT,SIGN_COLORS,INK,fmt,compact,pct,useWidth,useVisiblePoll,useTip,TipRow,Legend} from './viz.jsx';
import {withRun} from './Provenance.jsx';

const api=async path=>{const r=await fetch('/api/'+path);if(!r.ok)throw Error(await r.text());return r.json()};
const name=n=>n.type||`bodyId ${n.body_id}`;

function Composition({title,totals,families}){
 const [tip,show,hide]=useTip();const entries=families.map(f=>({...f,value:totals?.by_family?.[f.key]||0})).filter(f=>f.value>0);const sum=entries.reduce((a,f)=>a+f.value,0);
 if(!sum)return null;
 return <div className="composition viz-wrap"><div className="composition-title"><strong>{title}</strong><span>{fmt(totals.connections)} neuronas · {compact(totals.synapses)} sinapsis</span></div>
  <div className="stack" role="img" aria-label={title}>{entries.map(f=><b key={f.key} style={{flexGrow:f.value,background:FAMILY_COLORS[f.key]}} tabIndex="0" onPointerMove={e=>show(e,<><TipRow color={FAMILY_COLORS[f.key]} value={pct(f.value/sum)} label={f.label}/><div className="tip-meta">{compact(f.value)} sinapsis</div></>)} onPointerLeave={hide}/>)}</div>
  <div className="stack-labels">{entries.filter(f=>f.value/sum>=.08).map(f=><span key={f.key}><i style={{background:FAMILY_COLORS[f.key]}}/>{FAMILY_SHORT[f.key]} {pct(f.value/sum,0)}</span>)}</div>
  {totals.excitatory_synapses!=null&&<div className="ei"><span><i className="line" style={{background:SIGN_COLORS.excitatory}}/>Excitatorias {pct(totals.excitatory_synapses/totals.synapses,0)}</span><span><i className="line" style={{background:SIGN_COLORS.inhibitory}}/>Inhibitorias {pct(totals.inhibitory_synapses/totals.synapses,0)}</span></div>}
  {tip}</div>}

export default function NeuronGraph({bodyId,onSelect,run}){
 const [graph,setGraph]=useState(null),[error,setError]=useState('');const [ref,width]=useWidth();const [tip,show,hide]=useTip();
 useVisiblePoll(async alive=>{try{const next=await api(withRun(`neuron/${bodyId}/graph?limit=8`,run));if(alive()){setGraph(next);setError('')}}catch(e){if(alive())setError(e.message)}},5000,[bodyId,run]);
 // The observed wrapper stays mounted across loading states so its width is always measured.
 if(error||!graph||graph.neuron.body_id!==bodyId)return <div ref={ref} className="viz-wrap neuron-graph"><div className="empty">{error||`Cargando conexiones de ${bodyId}…`}</div></div>;
 const {neuron,inputs,outputs}=graph,vertical=width>0&&width<620,maxSyn=Math.max(1,...inputs.map(x=>x.synapses),...outputs.map(x=>x.synapses));
 const stroke=s=>1.2+9*Math.sqrt(s/maxSyn),radius=s=>5+7*Math.sqrt(s/maxSyn);
 let W,H,center,inPos,outPos;
 if(!vertical){const rows=Math.max(inputs.length,outputs.length,1),rowH=44;W=900;H=Math.max(260,rows*rowH+40);center=[450,H/2];
  inPos=inputs.map((_,k)=>[250,20+rowH/2+k*rowH+(rows-inputs.length)*rowH/2]);outPos=outputs.map((_,k)=>[650,20+rowH/2+k*rowH+(rows-outputs.length)*rowH/2])}
 else{const rowH=38;W=Math.max(300,width);H=inputs.length*rowH+outputs.length*rowH+210;center=[W/2,inputs.length*rowH+80];
  inPos=inputs.map((_,k)=>[24,34+k*rowH]);outPos=outputs.map((_,k)=>[W-24,center[1]+120+k*rowH])}
 const curve=([x0,y0],[x1,y1])=>vertical?`M${x0},${y0} C${x0},${(y0+y1)/2} ${x1},${(y0+y1)/2} ${x1},${y1}`:`M${x0},${y0} C${(x0+x1)/2},${y0} ${(x0+x1)/2},${y1} ${x1},${y1}`;
 const partnerTip=(e,p,direction)=>show(e,<><TipRow color={FAMILY_COLORS[p.family]} value={`${fmt(p.synapses)} sinapsis`} label={`${direction==='in'?'desde':'hacia'} ${name(p)}`}/><TipRow color={p.sign<0?SIGN_COLORS.inhibitory:SIGN_COLORS.excitatory} line value={p.sign<0?'Inhibitoria':'Excitatoria'} label="signo supuesto"/><div className="tip-meta">bodyId {p.body_id} · {graph.families.find(f=>f.key===p.family)?.label} · {p.spikes==null?'actividad no disponible':`${p.spikes} impulsos en 50 ms`} · toca para abrir</div></>);
 const node=(p,[x,y],direction)=>{const anchorEnd=vertical?direction==='out':direction==='in';return <g key={direction+p.body_id} className="graph-node" role="button" tabIndex="0" aria-label={`Abrir ${name(p)}`} onClick={()=>onSelect(p.body_id)} onKeyDown={e=>{if(e.key==='Enter')onSelect(p.body_id)}} onPointerMove={e=>partnerTip(e,p,direction)} onPointerLeave={hide}>
  <circle cx={x} cy={y} r="18" fill="transparent"/>{p.spikes>0&&<circle cx={x} cy={y} r={radius(p.synapses)+5} fill={FAMILY_COLORS[p.family]} opacity=".22"/>}<circle cx={x} cy={y} r={radius(p.synapses)} fill={FAMILY_COLORS[p.family]} stroke={INK.surface} strokeWidth="2"/>
  <text x={x+(anchorEnd?-16:16)} y={y-1} textAnchor={anchorEnd?'end':'start'}>{name(p)}</text><text className="value" x={x+(anchorEnd?-16:16)} y={y+12} textAnchor={anchorEnd?'end':'start'}>{fmt(p.synapses)} sin.{p.spikes>0?` · ${p.spikes} imp.`:''}</text></g>};
 return <div ref={ref} className="viz-wrap neuron-graph">
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Principales entradas y salidas de ${name(neuron)}`}>
   {!vertical&&<><text className="stage-title" x="250" y="14" textAnchor="middle">Principales entradas</text><text className="stage-title" x="650" y="14" textAnchor="middle">Principales salidas</text></>}
   {inputs.map((p,k)=><path key={'ei'+p.body_id} d={curve(inPos[k],center)} fill="none" stroke={p.sign<0?SIGN_COLORS.inhibitory:SIGN_COLORS.excitatory} strokeOpacity={vertical?.3:.5} strokeWidth={stroke(p.synapses)}/>)}
   {outputs.map((p,k)=><path key={'eo'+p.body_id} d={curve(center,outPos[k])} fill="none" stroke={neuron.sign<0?SIGN_COLORS.inhibitory:SIGN_COLORS.excitatory} strokeOpacity={vertical?.3:.5} strokeWidth={stroke(p.synapses)}/>)}
   {inputs.map((p,k)=>node(p,inPos[k],'in'))}{outputs.map((p,k)=>node(p,outPos[k],'out'))}
   <g onPointerMove={e=>show(e,<><TipRow color={FAMILY_COLORS[neuron.family]} value={name(neuron)} label={neuron.superclass||''}/><div className="tip-meta">bodyId {neuron.body_id} · {neuron.spikes==null?'actividad no disponible':`${neuron.spikes} impulsos en 50 ms`}</div></>)} onPointerLeave={hide}>
    {neuron.spikes>0&&<circle cx={center[0]} cy={center[1]} r="34" fill={FAMILY_COLORS[neuron.family]} opacity=".18"/>}
    <circle cx={center[0]} cy={center[1]} r="24" fill={FAMILY_COLORS[neuron.family]} stroke={INK.surface} strokeWidth="3"/>
    <text className="center-label" x={center[0]} y={center[1]+44} textAnchor="middle">{name(neuron)}</text>
    <text className="value" x={center[0]} y={center[1]+58} textAnchor="middle">{neuron.spikes==null?'—':`${neuron.spikes} impulsos / 50 ms`}</text>
   </g>
   {vertical&&<><text className="stage-title" x="24" y="10">Principales entradas ↓</text><text className="stage-title" x={W-24} y={center[1]+92} textAnchor="end">Principales salidas ↓</text></>}
  </svg>{tip}
  <Legend items={[{label:'Excitatoria (supuesta)',color:SIGN_COLORS.excitatory,line:true},{label:'Inhibitoria (supuesta)',color:SIGN_COLORS.inhibitory,line:true},...graph.families.map(f=>({label:f.label,color:FAMILY_COLORS[f.key]}))]}/>
  <p className="graph-note">Grosor y tamaño = número de sinapsis. Halo = disparó en la última ventana de 50 ms. Se muestran las 8 conexiones más fuertes de cada lado.</p>
  <div className="composition-grid"><Composition title="De dónde recibe" totals={graph.input_totals} families={graph.families}/><Composition title="A dónde envía" totals={graph.output_totals} families={graph.families}/></div>
 </div>}
