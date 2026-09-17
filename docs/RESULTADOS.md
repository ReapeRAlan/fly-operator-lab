# Resultados medidos por versión

Todas las cifras salen de archivos de `outputs/` o de la base local `work/learning/experiments.sqlite`, que no se publica. No hay estimaciones.

## Resumen

| Versión | Qué cambió | Resultado principal |
|---|---|---|
| v1 (15-sep) | MaleCNS completo en LIF controlando el juego, sin aprendizaje | Integración técnica validada ([README histórico](PROTOTIPO_V1_V2.md)) |
| v2 | Primer aprendizaje: PPO + R-STDP global | 88.8–94.3 % de los ejemplos eran esperas; no hubo aprendizaje útil |
| v3.1 (16-sep) | Macroacciones, currículo de 14 habilidades, DAgger por habilidad, R-STDP causal motora | En 3.75 h no salió de `move`; tick plástico de 6 s |
| v3.2 (16–17-sep) | Rendimiento ×13, interfaz sensorial nueva, plasticidad con signo | `move` resuelta por imitación; el adaptador PPO colapsó |

## v3.1: auditoría (16-sep)

Medido sobre la campaña `pilot-v3.1-curriculum` antes de detenerla:
- **Rendimiento:** tick de 0.25 s en frozen/adapter y **6.0–6.2 s** en internal. Se etiquetaban ~3.6 M aristas cuando la regla solo usa las 566,762 que entran a las 1,314 DN (2.2 %).
- **Plasticidad:** cambio máximo por actualización ~9e-6. El **37.4 %** de las aristas del cono motor son inhibitorias, y la regla v1 las movía en sentido contrario al crédito.
- **Actividad:** ~100 K espigas por tick de 50 ms; ~10 % de las neuronas activas.
- **Práctica autónoma:** mayoría de episodios cortados por `no_progress`; la entropía del actor rondaba 1.8 nats.

## v3.2: cambios validados

| Validación | v3.1 | v3.2 | Evidencia |
|---|---|---|---|
| Equivalencia bit a bit de la elegibilidad restringida (grafo real, 10 ticks) | — | **aprobada** | `outputs/motor_cone_equivalence.json` |
| Tick plástico (mediana) | 5.92 s | **0.44 s (13.4×)** | ídem |
| Aristas rastreadas al final | 3,669,214 | 144,959 | ídem |
| Sonda: sensores → acción (±45°) | 99.1 % | 99.1 % | `outputs/decodability_probe_v31.json`, `..._v32.json` |
| Sonda: DN → acción (±45°) | 78.8 % | **99.1 %** | ídem |
| Sonda: DN → acción (exacta) | 39.5 % | **78.8 %** | ídem |
| Transferencia pareada: DN cambiadas por sector del objetivo | 82–173 | **161–350** | `outputs/sensory_transfer_v3.json`, `..._v32.json` |
| Imitación v3.1 sobre features (exacta / ±45°) | 39.0 % / 78.7 % | **82.1 % / 98.6 %** | `outputs/imitation_offline_v31.json`, `..._v32.json` |
| Imitación alternativa v3.2 (exacta / ±45°) | 30.0 % / 66.9 % | 71.7 % / 97.9 % | ídem: **rechazada**, se conserva la v3.1 |
| Pruebas automatizadas | 74 | 91 | `outputs/v3_test_validation.json` |

Las sondas repiten en lazo abierto 63 episodios únicos de `move` (1,098 decisiones) con CV ridge agrupada por episodio.

### Calibración de la plasticidad (negativa)

Prueba pareada con los mismos episodios y semillas, acreditando la acción del instructor con errores de predicción reales (`outputs/motor_plasticity_calibration_v32.json`):

| Regla | Tasa | Δ log π acreditada | Decisiones mejoradas | Cambio de ganancia por episodio |
|---|---:|---:|---:|---:|
| v1 (sin signo) | 1e-4 | +0.066 | 50.7 % | 2.2e-6 |
| v2 (con signo) | 1e-4 | +0.007 | 51.2 % | 2.1e-6 |
| v1 | 1e-3 | −0.013 | 52.2 % | 1.9e-5 |
| v2 | 1e-3 | +0.004 | 50.2 % | 2.3e-5 |
| v1 | 1e-2 | −0.158 | 50.7 % | 1.9e-4 |
| v2 | 1e-2 | −0.420 | 37.8 % | 2.9e-4 |

Ninguna tasa cumplió el criterio preregistrado:
- **Tasas bajas:** el efecto es ruido.
- **Tasas altas:** mover eficacias desplaza las features descendentes y degrada al actor.

Se conservó la tasa 1e-4 y `internal` quedó como control.

## v3.2: noche del 16 al 17 de septiembre

- **Duración:** 19:28 → 07:00 (11.5 h), 96,255 ticks, 91 rondas del calendario.
- **Incidencias:**
  - 19:38: se cerró la ventana del juego (salida limpia).
  - 19:39: el juego relanzado crasheó por un `RestartMap` durante su carga inicial; se corrigió en `src/bridge_client.py`.
  - 19:56: reinicio controlado para bajar el mínimo de RAM a 1 GB.
  - Desde entonces, 0 fallos.
- **Enseñanza común:** 12/12 episodios exitosos por semilla; evaluación inicial 5/5 en las 3 semillas, al primer intento.

### Práctica autónoma (`move`): éxitos por cuarto de la noche

| Condición | Semilla 7 | Semilla 19 | Semilla 43 |
|---|---|---|---|
| frozen | 23/23 · 23/23 · 23/23 · 23/23 | 23/23 · 23/23 · 23/23 · 23/23 | 23/23 · 23/23 · 23/23 · 23/23 |
| internal | 23/23 · 23/23 · 23/23 · 23/23 | 23/23 · 23/23 · 23/23 · 23/23 | 22/22 · 22/22 · 22/22 · 25/25 |
| combined | 23/23 · 23/23 · 23/23 · 22/23 | 22/23 · 21/23 · 22/23 · 20/23 | 22/22 · 21/22 · 22/22 · 25/25 |
| **adapter** | **23/23 · 10/23 · 0/23 · 6/23** | **22/23 · 10/23 · 0/23 · 0/23** | **22/23 · 23/23 · 8/23 · 0/23** |

### Evaluación de promoción (5 episodios de validación, determinista)

| | Semilla 7 | Semilla 19 | Semilla 43 |
|---|---|---|---|
| frozen | 5/5 | 5/5 | 5/5 |
| internal | 5/5 | 5/5 | 5/5 |
| combined | 5/5 | 5/5 | 5/5 |
| **adapter** | 5/5 | **0/5** | **0/5** |

Resultado: la puerta exige ≥ 80 % en adapter, internal y combined para las 3 semillas, así que **no se promovió** y el currículo terminó en `move`.

### Lectura

- **Un 100 % sin aprendizaje autónomo:** `frozen` e `internal`, que no cambian el actor, logran el 100 %. El éxito en `move` viene de la interfaz v3.2 más la imitación, no de aprendizaje autónomo.
- **Colapso del adaptador:**
  - su entropía media reciente bajó a 0.04–0.18 (frozen: ~0.6);
  - sus decisiones por episodio subieron de ~10 a 35–47;
  - falla por `no_progress`.
- **Causas probables**, alineadas con la literatura (ver [INVESTIGACION_REFERENCIAS.md](INVESTIGACION_REFERENCIAS.md)):
  - actualización por episodio con 7–40 muestras;
  - normalización de la ventaja en lotes diminutos;
  - crítico sin calentar;
  - lr 3e-4;
  - ningún ancla al clon.
- **Siguiente paso:** [PLAN_MEJORA_V4.md](PLAN_MEJORA_V4.md).
