// Browser plugin not available: cua IAB reported unavailable; inventory had no browsers.
import {chromium} from '@playwright/test';import fs from 'node:fs';
const directory='D:/FlyOperatorLab/work/design';fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM?{executablePath:process.env.PLAYWRIGHT_CHROMIUM}:{})});const page=await browser.newPage({viewport:{width:1536,height:1024},deviceScaleFactor:1});let errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('console',msg=>{if(msg.type()==='error')errors.push(msg.text())});
await page.goto('http://127.0.0.1:8766');await page.getByRole('heading',{name:'Observatorio neuronal'}).waitFor();await page.getByRole('heading',{name:'Estado de la simulación'}).waitFor();
await page.screenshot({path:directory+'/panel_desktop.png'});
for(const name of ['Percepción','Actividad','Conectividad','Decisión']){await page.getByRole('button',{name,exact:true}).click();await page.screenshot({path:directory+'/panel_'+name.normalize('NFD').replace(/[\u0300-\u036f]/g,'')+'.png'});}
await page.getByRole('button',{name:'Conectividad',exact:true}).click();await page.getByRole('textbox',{name:'Buscar neurona'}).fill('10001');await page.getByRole('button',{name:'Buscar',exact:true}).click();await page.getByRole('button',{name:/10001 DNp01/}).first().click();await page.getByRole('heading',{name:/DNp01/}).waitFor();await page.getByLabel('Dirección de conexiones').selectOption('in');await page.getByText(/conexiones entrantes/).waitFor();await page.screenshot({path:directory+'/panel_connectivity_detail.png'});
await page.getByRole('button',{name:'Aprendizaje',exact:true}).click();await page.getByRole('textbox',{name:'Filtrar parámetros'}).fill('learning_rate');await page.getByRole('cell',{name:'dynamics.learning_rate',exact:true}).waitFor();
await page.setViewportSize({width:390,height:844});await page.screenshot({path:directory+'/panel_mobile.png'});
const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(overflow)errors.push('Mobile horizontal overflow');
fs.writeFileSync(directory+'/qa.json',JSON.stringify({url:page.url(),title:await page.title(),browser:'Playwright headless; Browser plugin not available',viewports:[[1536,1024],[390,844]],errors,passed:errors.length===0,interactions:['all five views','bodyId search','incoming connections','parameter filter']},null,2));
await browser.close();if(errors.length)throw Error(JSON.stringify(errors));
