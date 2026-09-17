import React,{useEffect,useRef,useState} from 'react';

// Categorical family slots in the validated fixed order (dataviz reference palette, light surface).
// "Motor y otras" folds the smallest superclasses into a neutral gray, keeping red free for inhibition.
export const FAMILY_COLORS={visual_projection:'#2a78d6',sensory:'#eb6834',optic:'#1baf7a',central:'#eda100',vnc:'#e87ba4',ascending:'#008300',descending:'#4a3aa7',other:'#8a9199'};
export const FAMILY_SHORT={visual_projection:'Proy. visual',sensory:'Sensorial',optic:'Óptico',central:'Central',vnc:'Cordón',ascending:'Ascend.',descending:'Descend.',other:'Otras'};
export const SIGN_COLORS={excitatory:'#2a78d6',inhibitory:'#e34948'};
export const ACCENT='#218e89';
export const INK={primary:'#233448',secondary:'#5d6f7e',muted:'#8b98a6',grid:'#e9eef1',axis:'#cfd8de',surface:'#ffffff',quiet:'#c7d0d6'};
export const BLUE_RAMP=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];

export const fmt=(x,d=0)=>x==null||Number.isNaN(x)?'—':Number(x).toLocaleString('es-MX',{maximumFractionDigits:d});
export function compact(x){if(x==null)return '—';const a=Math.abs(x);if(a>=1e6)return fmt(x/1e6,a>=1e7?0:1)+' M';if(a>=1e3)return fmt(x/1e3,a>=1e4?0:1)+' K';return fmt(x,a<10?1:0)}
export const pct=(x,d=1)=>x==null?'—':fmt(x*100,d)+' %';

export function useWidth(){const ref=useRef(null),[width,setWidth]=useState(0);useEffect(()=>{if(!ref.current)return;const o=new ResizeObserver(([e])=>setWidth(e.contentRect.width));o.observe(ref.current);return()=>o.disconnect()},[]);return [ref,width]}

// Poll only while the page is visible (phones lock, tabs hide) to spare the training laptop.
export function useVisiblePoll(callback,interval,deps=[]){useEffect(()=>{let live=true,timer;const run=async()=>{if(!document.hidden){try{await callback(()=>live)}catch{}}if(live)timer=setTimeout(run,interval)};const wake=()=>{if(!document.hidden){clearTimeout(timer);run()}};run();document.addEventListener('visibilitychange',wake);return()=>{live=false;clearTimeout(timer);document.removeEventListener('visibilitychange',wake)}},deps)}

// One floating tooltip per chart; content is React elements (never HTML strings).
export function useTip(){const [tip,setTip]=useState(null);const show=(event,content)=>{const x=event.clientX,y=event.clientY;if(x==null||y==null)return;setTip({x,y,content})};const hide=()=>setTip(null);useEffect(()=>{if(!tip)return;const clear=()=>setTip(null);window.addEventListener('scroll',clear,true);return()=>window.removeEventListener('scroll',clear,true)},[tip]);const node=tip&&<div className="viz-tip" role="status" style={{left:Math.max(8,Math.min(tip.x+14,window.innerWidth-212)),top:tip.y>window.innerHeight-150?tip.y-110:tip.y+16}}>{tip.content}</div>;return [node,show,hide]}

export function TipRow({value,label,color,line}){return <div className="tip-row">{color&&<i className={line?'line':''} style={{background:color}}/>}<strong>{value}</strong><span>{label}</span></div>}

export function Sparkline({values,width=120,height=30}){const v=values.filter(x=>x!=null);if(v.length<2)return <svg className="spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true"/>;const lo=Math.min(...v),hi=Math.max(...v),span=hi-lo||1;const pts=v.map((y,i)=>[i*(width-4)/(v.length-1)+2,height-3-(y-lo)/span*(height-6)]);return <svg className="spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true"><polyline points={pts.map(p=>p.join(',')).join(' ')} fill="none" stroke={INK.quiet} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"/><circle cx={pts.at(-1)[0]} cy={pts.at(-1)[1]} r="3" fill={ACCENT} stroke={INK.surface} strokeWidth="1.5"/></svg>}

function niceTicks(lo,hi,count=4){const span=hi-lo||Math.abs(hi)||1;const raw=span/count,mag=10**Math.floor(Math.log10(raw)),step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw);const start=Math.floor(lo/step)*step,ticks=[];for(let t=start;t<=hi+step*.5;t+=step)ticks.push(+t.toFixed(10));return ticks}

// Single-series line with a crosshair that snaps to the nearest step; no legend (the title names it).
export function LineChart({points,label,format=compact,xLabel=p=>`Paso ${p.id}`,height=210,zeroBased=false}){
 const [ref,width]=useWidth();const [tip,show,hide]=useTip();const [hover,setHover]=useState(null);
 const data=points.filter(p=>p.y!=null);const W=Math.max(width,280),H=height,L=52,R=14,T=14,B=28;
 if(data.length<2)return <div ref={ref} className="viz-wrap"><div className="empty">Esperando pasos registrados</div></div>;
 const ys=data.map(p=>p.y);let lo=zeroBased?0:Math.min(...ys),hi=Math.max(...ys);if(lo===hi){lo-=1;hi+=1}const ticks=niceTicks(lo,hi);lo=Math.min(lo,ticks[0]);hi=Math.max(hi,ticks.at(-1));
 const x=i=>L+i*(W-L-R)/(data.length-1),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 const move=e=>{const box=e.currentTarget.getBoundingClientRect();const i=Math.max(0,Math.min(data.length-1,Math.round((e.clientX-box.left-L)/((W-L-R)/(data.length-1)))));setHover(i);show(e,<><TipRow value={format(data[i].y)} label={label}/><div className="tip-meta">{xLabel(data[i])}</div></>)};
 return <div ref={ref} className="viz-wrap"><svg className="line-chart" width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onPointerMove={move} onPointerDown={move} onPointerLeave={()=>{setHover(null);hide()}}>
  {ticks.map(t=><g key={t}><line x1={L} x2={W-R} y1={y(t)} y2={y(t)} stroke={INK.grid}/><text x={L-8} y={y(t)+3} textAnchor="end">{format(t)}</text></g>)}
  <line x1={L} x2={W-R} y1={H-B} y2={H-B} stroke={INK.axis}/>
  <path d={`M${x(0)},${H-B} `+data.map((p,i)=>`L${x(i)},${y(p.y)}`).join(' ')+` L${x(data.length-1)},${H-B} Z`} fill={ACCENT} opacity=".08"/>
  <polyline points={data.map((p,i)=>`${x(i)},${y(p.y)}`).join(' ')} fill="none" stroke={ACCENT} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"/>
  <text x={L} y={H-8}>{xLabel(data[0])}</text><text x={W-R} y={H-8} textAnchor="end">{xLabel(data.at(-1))}</text>
  {hover!=null&&<g><line x1={x(hover)} x2={x(hover)} y1={T} y2={H-B} stroke={INK.muted}/><circle cx={x(hover)} cy={y(data[hover].y)} r="4.5" fill={ACCENT} stroke={INK.surface} strokeWidth="2"/></g>}
 </svg>{tip}</div>}

export function Legend({items}){return <div className="viz-legend">{items.map(item=><span key={item.label}><i className={item.line?'line':''} style={{background:item.color}}/>{item.label}</span>)}</div>}
