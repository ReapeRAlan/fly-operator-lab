// Visual QA for the campaign selector, provenance labels and stale-data safeguards.
import { chromium } from '@playwright/test';
import fs from 'node:fs';

const directory = 'D:/FlyOperatorLab/work/design';
const url = 'http://127.0.0.1:8766';
fs.mkdirSync(directory, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM } : {}),
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1024 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
page.on('response', response => { if (response.status() === 404) errors.push('404 ' + response.url()); });

async function expectText(text, label) {
  if (!await page.getByText(text).count()) errors.push('Missing: ' + label);
}
async function tab(name, image, settle = 1200) {
  await page.getByRole('button', { name, exact: true }).click();
  await page.waitForTimeout(settle);
  await page.screenshot({ path: directory + '/' + image });
}

try {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: 'Observatorio neuronal' }).waitFor({ timeoutMs: 5000 });
  const selector = page.getByLabel('Seleccionar campaña');
  await selector.waitFor({ timeoutMs: 5000 });
  await page.waitForTimeout(2500);
  await expectText(/Obsoleto|Sin datos|En vivo/, 'provenance');
  await page.screenshot({ path: directory + '/v41_desktop_overview.png' });

  await tab('Salud', 'v41_desktop_health.png');
  await expectText('Salud del observatorio', 'health panel');
  await tab('Plasticidad', 'v41_desktop_plasticity.png');
  await expectText(/Auditoría no archivada|Revisión/, 'plasticity archive state');
  await tab('Circuito', 'v41_desktop_circuit.png', 1200);
  await expectText(/Circuito anatómico y última decisión registrada|Del juego a la acción, en vivo/, 'circuit context');
  await tab('Percepción', 'v41_desktop_perception.png');
  await expectText(/Historial de acciones|Aún no hay observaciones/, 'perception');
  await tab('Cerebro 3D', 'v41_desktop_brain3d.png', 2500);
  await expectText('Modelo cerebral 3D', 'brain 3D');

  let values = [];
  for (let attempt = 0; attempt < 12 && values.length < 2; attempt += 1) {
    values = await selector.locator('option').evaluateAll(options => options.map(option => option.value));
    if (values.length < 2) await page.waitForTimeout(500);
  }
  const historical = values.find(value => value !== 'v4.1-night1');
  if (!historical) { errors.push('No historical campaign was available for visual QA'); throw Error('No historical campaign was available for visual QA'); }
  await page.goto(url + '?run=' + encodeURIComponent(historical), { waitUntil: 'domcontentloaded' });
  await page.getByLabel('Seleccionar campaña').waitFor({ timeoutMs: 5000 });
  await page.waitForTimeout(2500);
  await expectText(/Histórico/, 'historical provenance');
  await tab('Circuito', 'v41_historical_circuit.png', 900);
  await expectText('Actividad neuronal no disponible para este contexto.', 'historical circuit safeguard');
  await tab('Cerebro 3D', 'v41_historical_brain3d.png', 2500);
  if (await page.getByText('Activa en la última ventana').count()) errors.push('Historical brain 3D exposed live activity legend');

  await page.setViewportSize({ width: 390, height: 844 });
  await tab('Salud', 'v41_mobile_health.png');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  if (overflow) errors.push('Mobile horizontal overflow');
  await page.screenshot({ path: directory + '/v41_mobile_historical.png', fullPage: true });
} finally {
  fs.writeFileSync(directory + '/qa_v41.json', JSON.stringify({
    url: page.url(),
    browser: 'Playwright headless with Chrome executable',
    viewports: [[1536, 1024], [390, 844]],
    checks: ['campaign selector', 'provenance', 'obsolete current state', 'historical read-only circuit', 'historical 3D without live legend', 'mobile overflow'],
    errors,
    passed: errors.length === 0,
  }, null, 2));
  await browser.close();
}
if (errors.length) throw Error(JSON.stringify(errors));
