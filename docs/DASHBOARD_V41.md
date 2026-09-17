# Dashboard v4.1: contexto verificable y campañas

El observatorio muestra una sola campaña por vez. La API resuelve un contexto común en cada lectura, con selected_run, configured_run, origen del snapshot, fecha, edad, coherencia y vigencia. La interfaz pinta ese contexto antes de cualquier panel científico.

## Vigencia

- **En vivo**: status.json, latest.json y live_version.json declaran el mismo run; la revisión es coherente, tiene menos de 10 segundos y el estado del entrenador permite publicar actividad.
- **Histórico**: se consulta otra campaña persistida. Es de solo lectura y usa SQLite más los archivos archivados de esa campaña.
- **Obsoleto**: la campaña configurada no tiene un snapshot vigente o los archivos vivos pertenecen a otro run.
- **Sin datos**: no hay campaña persistida ni snapshot vigente.

Un archivo vivo sin run, con un run distinto o con más de 10 segundos no se presenta como telemetría actual.

## Rutas relevantes

- GET /api/campaigns: campañas configurada y persistidas, episodios, pasos y última actividad.
- Todas las vistas dependientes de campaña aceptan run: estado, historial, observatorio, circuito, cerebro 3D, plasticidad, neurona, equipo y exportaciones.
- GET /api/plasticity?run=... solo lee work/learning/campaigns/<run>/gain_audit.jsonl. El histórico global ya no se reutiliza.
- Las exportaciones de evidencia incluyen context.json y no adjuntan los archivos vivos de otra campaña.

## Vistas

Salud, Circuito, Plasticidad, Cerebro 3D, Percepción y Decisión reciben una banda de procedencia común. Circuito y Cerebro 3D conservan anatomía y la última decisión durable en consulta histórica, pero ocultan tasas, gains, trazas y controles que no existen para esa campaña.

Cerebro 3D inicia con 800 neuronas en pantallas de hasta 720 px. Su escena renderiza al cargar, al redimensionar y al interactuar; no mantiene un ciclo de render continuo cuando la vista no está activa.

## Validación

Ejecutar desde la raíz:

    .\.venv-learning\Scripts\python.exe -m pytest -q
    cd dashboard
    npm run build
    $env:PLAYWRIGHT_CHROMIUM='C:\Program Files\Google\Chrome\Application\chrome.exe'
    npm run qa:visual

La QA visual guarda capturas y work/design/qa_v41.json. Comprueba el selector, procedencia, estado obsoleto, campaña histórica, Cerebro 3D sin actividad archivada y el ancho de 390 px.
