# Fly Operator v3.2: rendimiento, interfaz sensorial y plasticidad corregida

Generado: 2026-09-16T19:32:46-06:00. Campaña: `pilot-v3.2-curriculum`. La campaña v3.1 se conserva intacta como control.

## 1. Rendimiento: elegibilidad restringida al cono motor

La regla causal motora solo acredita aristas que entran en las 1,314 descendentes (566,762 aristas). El integrador ya no etiqueta el resto del grafo; la dinámica de las 166,700 neuronas y 25,582,938 aristas no cambia.

- Equivalencia bit a bit sobre el grafo real (`work\learning\campaigns\pilot-v3.1-curriculum\checkpoints\internal_7\current.json`, 10 ticks): **aprobada**. Incluye voltajes, corrientes, refractarios, cola de retardo, trazas, ganancias, spikes por tick, instantáneas de elegibilidad motora y actualizaciones.
- Tick plástico: 5.92 s → **0.44 s** (mediana; **13.4×**). Aristas rastreadas al final: 3,669,214 → 144,959.

## 2. Interfaz sensorial v3.2

- Mismos 98 canales, mismas 1,568 compuertas excitatorias y mismo ranking anatómico. Cambia el **orden de asignación**: dirección y proximidad del objetivo reciben las rutas visuales más fuertes y el estado dinámico las rutas corporales más fuertes. En v3.1, el objetivo recibía rutas débiles y health/ammo, casi constantes, las más fuertes.
- Sectores con afinación von Mises (κ=2.0); objetivo siempre saliente; `goal_distance` codifica proximidad exp(−d/1.5 m); ganancia baja para estados de variación lenta.

**Sonda de decodificación** (repetición en lazo abierto de episodios `move` grabados; CV ridge agrupada por episodio con penalización elegida por CV interna; trayectorias duplicadas excluidas):

| Esquema | Episodios | Decisiones | Sensores → acción (±45°) | DN → acción exacto | DN → acción ±45° | Spikes / 50 ms |
|---|---:|---:|---:|---:|---:|---:|
| v3.1 | 63 | 1098 | 99.1 % | 39.5 % | 78.8 % | 101,260 |
| v3.2 | 63 | 1098 | 99.1 % | 78.8 % | 99.1 % | 99,879 |

- Criterio preregistrado DN → acción ±45° ≥ 90 %: **cumplido**.
- Fracción de DN ≥360 Hz: v3.1 0.2 %, v3.2 0.2 %. No bajó; ya era marginal y no limitaba la decodificación.
- Transferencia pareada: v3.1 8/8 firmas, 82–173 DN cambiadas por sector; v3.2 **aprobada**, 8/8 firmas, **161–350** DN cambiadas.

Límite: la sonda mide señal disponible para un lector lineal sobre trayectorias grabadas; no mide competencia en lazo cerrado.

## 3. Actor por imitación

| Features | Procedimiento v3.1 exacto / ±45° | Procedimiento v3.2 exacto / ±45° |
|---|---:|---:|
| v3.1 | 39.0 % / 78.7 % | 30.0 % / 66.9 % |
| v3.2 | 82.1 % / 98.6 % | 71.7 % / 97.9 % |

La hipótesis de subajuste del actor fue **refutada**: el procedimiento v3.1 alcanza el techo lineal. El optimizador separado con parada temprana generalizó peor y **no se adopta** (el código queda disponible, desactivado). La mejora del actor proviene de la interfaz sensorial.

## 4. Plasticidad causal motora v2

`causal_motor_rstdp_v1` movía cada ganancia en la dirección del crédito sin considerar el signo presináptico; en aristas inhibitorias (37.4 % del cono motor) eso reduce la actividad de la descendente acreditada. `causal_motor_rstdp_v2` multiplica el cambio por el signo presináptico; el signo y la anatomía no cambian.

Calibración pareada (mismos episodios y semillas; crédito positivo a la acción del instructor con magnitudes de error registradas en v3.1):

| Regla | Tasa | Δ log π acreditada | Decisiones mejoradas | Cambio medio de ganancia por episodio | En límites |
|---|---:|---:|---:|---:|---:|
| `causal_motor_rstdp_v1` | 0.0001 | +0.0658 | 50.7 % | 2.18e-06 | 0.0 % |
| `causal_motor_rstdp_v2` | 0.0001 | +0.0071 | 51.2 % | 2.13e-06 | 0.0 % |
| `causal_motor_rstdp_v1` | 0.001 | -0.0128 | 52.2 % | 1.93e-05 | 0.0 % |
| `causal_motor_rstdp_v2` | 0.001 | +0.0038 | 50.2 % | 2.32e-05 | 0.0 % |
| `causal_motor_rstdp_v1` | 0.01 | -0.1580 | 50.7 % | 1.90e-04 | 0.0 % |
| `causal_motor_rstdp_v2` | 0.01 | -0.4201 | 37.8 % | 2.94e-04 | 0.0 % |

Tasa seleccionada por el criterio preregistrado: **None**. Tasa configurada: **0.0001**. Criterio: choose the smallest rate whose v2 mean change per episode is 1e-3..1e-2, <1% of motor-cone gains at bounds, and mean delta log pi(credited) > 0.

**Resultado negativo.** Ninguna tasa cumplió el criterio. Con tasas bajas, los cambios de ganancia son minúsculos y el cambio de log π acreditado es indistinguible del azar (~50 % de decisiones mejoradas). Con 1e-2, la política empeora en ambas reglas: mover eficacias desplaza las features descendentes fuera de la distribución con la que se ajustó el actor, y ese efecto domina al crédito. En lazo abierto la prueba no distingue v2 de v1; v2 se mantiene porque su dirección es correcta por construcción (prueba unitaria sobre una sinapsis aislada). Se conserva la tasa v3.1 (la menos disruptiva). La condición `internal` sigue en el protocolo como control; no se espera aprendizaje atribuible a ella con esta regla.

## 5. Operación

- Plazo del piloto persistente en `schedule.json`; un reinicio no lo extiende (`-ResetDeadline` lo reinicia explícitamente). Plazo actual: 2026-09-17T07:00.
- RAM con histéresis: pausa < 1.5 GB, reanuda ≥ 2.0 GB.
- Supervisor (`scripts/supervise_learning.ps1`, iniciado por `start_learning.ps1`): si el entrenador termina sin un paro intencional antes del plazo, lo relanza (máximo 6 veces) desde el último checkpoint, conserva los registros del fallo como `work/learning/crash_*` y anota en `work/learning/supervisor.log`. `faulthandler` registra fallos nativos.
- Capturas v1/v2 (3.3 GiB) archivadas fuera de la cuota en `archive/frames` con manifiesto SHA-256 (`scripts/archive_frames.py --restore` las devuelve).
- SQLite ya no repite las 100 etiquetas del catálogo por paso; el historial las reconstruye. El hash del actor se calcula una vez por episodio.
- Pruebas automatizadas: **88**, aprobadas: True.

## 6. Resultados de la campaña v3.2

| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95 % Wilson |
|---|---:|---|---|---:|---:|---|
| common_teaching | 7 | move | train | 12 | 12 | 75.8%–100.0% |

## 7. Límites

- GPU: no se adoptó. El integrador por eventos procesa ~200 spikes por paso de 0.1 ms; una versión GPU requeriría un kernel CUDA completo, una nueva validación y perdería reproducibilidad bit a bit con los checkpoints CPU.
- Las sondas son de lazo abierto; la competencia autónoma solo se afirma con la validación reservada del currículo.
- La dinámica LIF, los signos de neurotransmisor y la interfaz son supuestos de ingeniería documentados, no fisiología validada.
