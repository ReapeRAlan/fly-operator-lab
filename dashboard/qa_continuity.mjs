// Browser plugin not available. Real local panel controls, existing headless Chromium.
import {chromium} from '@playwright/test';import fs from 'node:fs';
const root='D:/FlyOperatorLab/work/design';const url='http://127.0.0.1:8766';
const state=async()=>{const r=await fetch(url+'/api/state');if(!r.ok)throw Error('state '+r.status);return r.json()};
const until=async check=>{const end=Date.now()+45000;while(Date.now()<end){const s=await state();if(check(s))return s;await new Promise(r=>setTimeout(r,500));}throw Error('Timed out waiting for trainer')};
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM?{executablePath:process.env.PLAYWRIGHT_CHROMIUM}:{})});
const page=await browser.newPage({viewport:{width:1536,height:1024}});const errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
try{
 await page.goto(url);await page.getByRole('heading',{name:'Observatorio neuronal'}).waitFor();
 const before=await until(s=>s.latest?.sequence>0&&['running','saving'].includes(s.status.state));
 await page.getByRole('button',{name:'Pausar',exact:true}).click();
 const paused=await until(s=>s.status.state==='paused'&&s.status.reason==='Pausa solicitada');
 await page.getByRole('button',{name:'Reanudar',exact:true}).waitFor();await page.screenshot({path:root+'/panel_pause_verified.png'});
 await new Promise(r=>setTimeout(r,1500));const still=await state();
 if(still.latest.sequence!==paused.latest.sequence||still.latest.episode!==paused.latest.episode)throw Error('Game advanced during manual pause');
 await page.getByRole('button',{name:'Reanudar',exact:true}).click();
 const resumed=await until(s=>s.status.state==='running'&&(s.latest.sequence!==paused.latest.sequence||s.latest.episode!==paused.latest.episode));
 await page.getByRole('button',{name:'Pausar',exact:true}).waitFor();await page.screenshot({path:root+'/panel_training_active.png'});
 fs.writeFileSync(root+'/qa_continuity.json',JSON.stringify({passed:errors.length===0,errors,before:{episode:before.latest.episode,sequence:before.latest.sequence},paused:{episode:paused.latest.episode,sequence:paused.latest.sequence},resumed:{episode:resumed.latest.episode,sequence:resumed.latest.sequence},checkpoint_rows:resumed.checkpoints.length},null,2));
 if(errors.length)throw Error(errors.join('\n'));
}finally{await browser.close()}
