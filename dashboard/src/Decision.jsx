import React from 'react';
import {ACCENT,INK,SIGN_COLORS,fmt,pct,useTip,TipRow,Legend} from './viz.jsx';

// Heading-up compass: engine angles grow from +x toward +z, which is clockwise on the X/Z map.
export function MoveCompass({latest}){
 const [tip,show,hide]=useTip();const d=latest?.decision;
 if(!d?.probabilities)return <div className="empty">Esperando una decisión</div>;
 const labels=d.action_labels||[],moves=labels.map((label,i)=>({label,i,p:d.probabilities[i],allowed:!!d.mask?.[i],angle:Number((label.match(/^move · (-?\d+)°/)||[])[1])})).filter(m=>!Number.isNaN(m.angle));
 const channels=latest.pre_action?.channels||{},hasGoal=channels.goal_bearing_sin!=null;
 const goalAngle=hasGoal?Math.atan2(2*channels.goal_bearing_sin-1,2*channels.goal_bearing_cos-1)*180/Math.PI:null;
 const allowedMoves=moves.filter(m=>m.allowed);const stop=labels.indexOf('stop');
 const S=300,c=S/2,R=112,maxP=Math.max(1e-9,...allowedMoves.map(m=>m.p));
 const polar=(deg,r)=>{const a=deg*Math.PI/180;return [c+r*Math.sin(a),c-r*Math.cos(a)]};
 const wedge=(deg,r)=>{const [x0,y0]=polar(deg-19,r),[x1,y1]=polar(deg+19,r);return `M${c},${c} L${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1} Z`};
 const executed=d.executed??d.selected;
 return <div className="compass-layout viz-wrap"><svg className="compass" viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Probabilidad de moverse en cada dirección respecto a hacia dónde mira el operador">
  {[.33,.66,1].map(k=><circle key={k} cx={c} cy={c} r={R*k} fill="none" stroke={INK.grid}/>)}
  {[0,45,90,135,180,225,270,315].map(a=>{const [x,y]=polar(a,R);return <line key={a} x1={c} y1={c} x2={x} y2={y} stroke={INK.grid}/>})}
  <text className="stage-title" x={c} y="14" textAnchor="middle">Frente</text>
  {moves.map(m=>{const r=m.allowed?12+(R-12)*Math.sqrt(m.p/maxP):0,isExec=executed===m.i;return <g key={m.i} tabIndex="0" onPointerMove={e=>show(e,<><TipRow value={pct(m.p)} label={`mover ${m.angle}°`}/><div className="tip-meta">{!m.allowed?'No disponible en este paso':isExec?'Dirección ejecutada':d.teacher_action===m.i?'Elección del instructor':'Probabilidad del actor'}</div></>)} onPointerLeave={hide}>
   <path d={wedge(m.angle,R)} fill="transparent"/>{m.allowed&&<path d={wedge(m.angle,r)} fill={isExec?ACCENT:INK.quiet} stroke={INK.surface} strokeWidth="2"/>}
   {m.allowed&&m.p/maxP>.35&&(()=>{const [x,y]=polar(m.angle,Math.min(R+16,r+14));return <text className="value" x={x} y={y+4} textAnchor="middle">{pct(m.p,0)}</text>})()}</g>})}
  {goalAngle!=null&&(()=>{const [x,y]=polar(goalAngle,R+8),[bx,by]=polar(goalAngle,R-26);return <g><line x1={bx} y1={by} x2={x} y2={y} stroke={INK.primary} strokeWidth="2" strokeLinecap="round"/><path d="M0,-7 L7,0 L0,7 L-7,0 Z" transform={`translate(${x},${y})`} fill={INK.surface} stroke={INK.primary} strokeWidth="2"/></g>})()}
  <circle cx={c} cy={c} r="9" fill={INK.primary}/><path d={`M${c},${c-18} L${c-6},${c-8} L${c+6},${c-8} Z`} fill={INK.primary}/>
 </svg>{tip}
 <div className="compass-side">
  <div><span>Orden ejecutada</span><strong>{latest.action_label||'—'}</strong><small>{d.controller==='instructor'?'Instructor':'Actor neuronal'}</small></div>
  <div><span>Objetivo</span><strong>{goalAngle==null?'—':`${fmt((Math.round(goalAngle)%360+360)%360)}°`}</strong><small>{latest.pre_action?.goal_distance==null?'':`a ${fmt(latest.pre_action.goal_distance,2)} m`}</small></div>
  <div><span>Detenerse</span><strong>{stop>=0&&d.mask?.[stop]?pct(d.probabilities[stop]):'No disponible'}</strong><small>probabilidad de stop</small></div>
  <div><span>Certeza</span><strong>{d.entropy==null?'—':fmt(Math.abs(d.entropy)<.005?0:d.entropy,2)}</strong><small>entropía (menor = más decidido)</small></div>
  <Legend items={[{label:'Dirección ejecutada',color:ACCENT},{label:'Otras direcciones',color:INK.quiet},{label:'Hacia el objetivo (rombo)',color:INK.primary,line:true}]}/>
 </div></div>}

export function ContributionBars({latest,onInspect}){
 const [tip,show,hide]=useTip();const rows=(latest?.decision?.contributions||[]).slice(0,14);
 if(!rows.length)return <div className="empty">Sin contribuciones registradas</div>;
 const max=Math.max(...rows.map(r=>Math.abs(r.logit_contribution)),1e-9);
 return <div className="viz-wrap contrib">{rows.map((r,k)=>{const share=Math.abs(r.logit_contribution)/max*50,positive=r.logit_contribution>=0;return <div className="contrib-row" key={k} tabIndex="0" onPointerMove={e=>show(e,<><TipRow color={positive?SIGN_COLORS.excitatory:SIGN_COLORS.inhibitory} value={fmt(r.logit_contribution,4)} label={positive?'a favor de la acción':'en contra de la acción'}/><div className="tip-meta">DN {r.body_id} · filtro {r.filter_ms} ms · actividad {fmt(r.feature,3)} × peso {fmt(r.weight,4)}</div></>)} onPointerLeave={hide}>
  <button className="text-button" onClick={()=>onInspect(r.body_id)}>{r.body_id}<small>{r.filter_ms} ms</small></button>
  <div className="diverge"><span className="mid"/><b style={{[positive?'left':'right']:'50%',width:share+'%',background:positive?SIGN_COLORS.excitatory:SIGN_COLORS.inhibitory}}/></div>
  <span className="num">{fmt(r.logit_contribution,3)}</span></div>})}
  <Legend items={[{label:'Empuja hacia la acción propuesta',color:SIGN_COLORS.excitatory},{label:'Empuja en contra',color:SIGN_COLORS.inhibitory}]}/>{tip}</div>}
