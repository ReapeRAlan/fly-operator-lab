# Fly Operator v3.1: aprendizaje causal, currículo y observación científica

**Dictamen:** la infraestructura está implementada y opera con el grafo clasificado completo. La transferencia sensorial y las reglas del instructor están comprobadas. La competencia autónoma y la generalización permanecen pendientes de evaluación reservada.

Informe generado: 2026-09-16T15:17:21-06:00. Campaña: `pilot-v3.1-curriculum`. El estado vivo se consulta en [el panel local](http://127.0.0.1:8766); no se congela como conclusión científica.

## 1. Inventario y procedencia

- **166,700** neuronas clasificadas y **25,582,938** aristas dirigidas, sin poda del grafo clasificado.
- Identidad, dirección y cantidad anatómica de cada arista permanecen inmutables. La eficacia aprendida se guarda aparte.
- Anatomía y anotaciones: MaleCNS v1.0. Signos de neurotransmisor, dinámica LIF y asignaciones del juego tienen supuestos documentados.
- Los segmentos sin `superclass` permanecen en la fuente y no se presentan como neuronas simuladas.

## 2. Interfaz sensorial v3

- 98 canales × 16 puertas = **1,568 puertos**, todos únicos: **1,568**.
- Espacio: puertas `visual_projection` excitatorias clasificadas por rutas anatómicas directas y a dos saltos hacia descendentes.
- Estado y tarea: puertas excitatorias `vnc_sensory` y `cb_sensory`.
- Señales explícitas para enemigos, amistosos, rehenes, civiles, puertas, objetivo, paredes, salud, munición, postura confirmada por el motor, equipo, acción en curso, dirección/distancia y tiempo restante.
- Prueba pareada de transferencia: **aprobada**; 8/8 firmas de objetivo distintas; rango observado de descendentes cambiadas por sector/semilla: **82–173**; saturación máxima medida: 0.0.

Este resultado demuestra que una señal llega al readout motor. No demuestra que el actor la interprete correctamente.

## 3. Tiempo, decisiones y currículo

- Los ticks con calentamiento, cola o recibo `in_progress` avanzan motor y cerebro, pero no crean demostraciones ni decisiones PPO.
- Una excepción explícita aparece solo cuando el contexto permite intervenir: `stop` cerca del objetivo o `cancel` durante una brecha. Esas intervenciones sí son decisiones y reciben crédito propio.
- Movimiento exige llegar y completar `stop`; postura exige que el bit nativo `crouched` quede activo; la cancelación se acepta solo si el motor la completa y la puerta permanece cerrada.
- Una macroacción acumula recompensa hasta quedar libre o terminar. El descuento de su bootstrap es `γ^k` para `k` ticks.
- PPO se actualiza al final del episodio natural; `n_steps=256` no corta la misión.
- Cada habilidad recibe 6 episodios guiados y 6 DAgger por semilla. Si falla el gate común, admite hasta 12 correcciones con probabilidad de instructor 0.25.
- Espera libre repetida se corta a 48 decisiones; navegación sin mejora se corta a 32 decisiones y queda rotulada como fallo/truncación.

## 4. Plasticidad causal motora

`causal_motor_rstdp_v1` congela, antes de la acción, la elegibilidad de aristas activas cuyo destino es una de las 1,314 neuronas descendentes. Al concluir la macroacción combina el error temporal con `d log π(a|s) / d DN` y decae la instantánea según el retraso real hasta el crédito. La consecuencia observada después de actuar no puede entrar retroactivamente en esa instantánea.

La condición interna mantiene fijo el actor; cambia el evaluador externo y la eficacia de esas aristas. La combinada alterna episodios de PPO e internos. Es una regla computacional experimental, no una medición de dopamina ni una afirmación fisiológica.

## 5. Evaluación y no contaminación

- Diagnóstico de promoción cada 512 decisiones cognitivas, con 5 episodios por habilidad adquirida.
- Gate: al menos 80% para adaptador, interna y combinada en semillas [7, 19, 43]; la referencia congelada se informa.
- Aceptación: 30 episodios de validación por semilla. `test` se reserva para el cierre.
- Antes de evaluar se guarda un checkpoint; después se restauran cerebro, actor, filtros, optimizador, contadores y RNG. Las pruebas automatizadas comprueban esa igualdad.
- La telemetría de validación se publica como bloque atómico. La limpieza auditada archivó 1481 pasos y 5 episodios parciales previos; quedan 0 episodios parciales activos.
- Éxito técnico, competencia y evidencia de aprendizaje se reportan por separado.

## 6. Checkpoints y recursos

- Calendario: `work\learning\campaigns\pilot-v3.1-curriculum\schedule.json`.
- Checkpoints: `work\learning\campaigns\pilot-v3.1-curriculum\checkpoints/<condición>_<semilla>/current.json`.
- Cada generación guarda cerebro, retardos, elegibilidad, eficacias, actor, evaluador, optimizador, filtros, ejemplos y RNG con hashes SHA-256.
- Antes de publicar el puntero se rechaza cualquier `NaN`/infinito en cerebro, política, optimizador, filtros o rasgos; el hash semántico fija dinámica, recompensa, currículo, escenarios, capacidades, catálogo, acciones y esquema sensorial.
- Contrato semántico de capacidades v2: separa acciones/evidencia de su PID y hash de compilación. Migración auditada del checkpoint: **aprobada**; los hashes de cerebro, política, ejemplos, actor y grafo permanecieron intactos.
- Perfil `performance_cpu_v1`: afinidad [2, 4, 6, 8], PyTorch 2 hilos, prioridad below_normal, pausa real 0.05 s y juego 20 fps.
- Política de pantalla operativa: `single_monitor`; pantalla despierta durante el entrenamiento: True. Esta política no altera los hashes científicos ni los checkpoints.
- Guardas: 1.5 GB RAM disponibles, 4 GB por worker, 8 GB libres y cuota 8 GB.

## 7. Verificación disponible

- Instructor nativo: **42/42** casos exitosos.
- Transferencia sensorial: **aprobada**.
- Pruebas automatizadas registradas: **True**; cantidad: 74.
- Parada y reanudación real verificadas por hash de actor y conteo exacto de ejemplos antes de continuar.
- Datos v3 al generar este informe: 2803 pasos persistidos y 33 episodios completos.

Evidencias: [transferencia sensorial](sensory_transfer_v3.json), [instructor](teacher_validation_v3.json), [configuración](../config/learning.json), [esquema sensorial](../data/learning/sensory_ports_v31.json).

## 8. Resultados registrados de la campaña v3

| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95% Wilson |
|---|---:|---|---|---:|---:|---|
| common_teaching | 7 | move | train | 13 | 13 | 77.2%–100.0% |
| common_teaching | 7 | move | validation | 10 | 8 | 49.0%–94.3% |
| common_teaching | 19 | move | train | 10 | 10 | 72.2%–100.0% |

Calendario al corte: etapa `0`, ronda `0`. Evaluaciones guardadas: 0. Estos valores describen progreso operativo y pueden cambiar mientras corre el entrenador.

## 9. Límites

- Los resultados v2 se conservan como control histórico y no se atribuyen a v3: cambiaron sensores, temporalidad, demostraciones y plasticidad.
- El modelo no incluye toda la fisiología de una mosca ni afirma equivalencia biológica.
- Las cinco familias nativas pendientes (`use`, `clear_obstacle`, `arrest`, `spy_camera`, `evacuate`) siguen fuera del entrenamiento hasta validar sus efectos.
- No se afirmará resolución autónoma de misiones hasta superar validación reservada en las tres semillas y comparación contra el punto común y la referencia congelada.

Hashes: esquema sensorial `21c0c7de03b8a323024c372fe1a364997bc9a318cca8469976d0a5b74e41b1a9`; prueba de transferencia `a36b3c66e266c60cfeea60563c33ca3b3cdda3e37046808a3710b9b5f37c8f78`.
