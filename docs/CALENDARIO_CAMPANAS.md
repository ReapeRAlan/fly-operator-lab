# Calendario de campañas v4.1

Guía de una página para lanzar y detener el entrenamiento. Todo se hace con doble clic; el
plazo lo calcula el lanzador desde el reloj del equipo.

## Lanzadores

| Doble clic en | Qué hace | Cuándo usarlo |
|---|---|---|
| **INICIAR NOCHE.cmd** | Entrena hasta las **07:00 del día siguiente** | Al llegar a casa entre semana |
| **INICIAR FIN DE SEMANA.cmd** | Entrena sin parar hasta el **lunes 07:00** | Viernes por la noche |
| **INICIAR APRENDIZAJE.cmd** | 12 horas desde ahora | Pruebas sueltas |

Los tres abren el juego si hace falta, arrancan el panel <http://127.0.0.1:8766> y dejan un
supervisor que reinicia al entrenador si se cae (sin mover el plazo). Para detener antes de
tiempo: botón **Detener** del panel.

## Plan de esta semana

| Ventana | Campaña | Contenido |
|---|---|---|
| Jue 17-sep ~19:00 → vie 07:00 | `v4.1-night1` | Diagnóstico: PPO anclado contra controles, etapas `move` → `switch` |
| Vie 18-sep 07:00–18:00 | — | Análisis de la noche, decisiones de armas, tarea con memoria |
| Vie 18-sep ~19:00 → lun 21-sep 07:00 | `v4.1-weekend` | Currículo completo + pista científica + examen final del lunes 05:30 |

## Qué corre dentro de una campaña

- **Pista principal** (`ppo_anchored`, 3 semillas): conectoma + imitación/DAgger + PPO anclado
  al instructor. Es la única que promueve de etapa, con **bloques fijos de 25 episodios de
  validación** y promoción con **≥ 24/25** más la retención de las etapas anteriores.
- **Pista científica** (~1/3 del tiempo, siempre en `move`): `frozen_bc` (sin RL),
  `adapter_half_updates` (mitad de actualizaciones), `sensor_only` (sin cerebro, mismos
  filtros) y `bias_only` (entrada constante). No promueve nada; sirve para comparar.
- Cada actualización de PPO pasa una **prueba de identidad** (cociente 1, KL 0) antes de
  aplicarse; si falla, no se actualiza y queda registrado.
- Si un bloque cae más de 20 puntos respecto al mejor, el aprendiz **vuelve** a su mejor
  checkpoint completo.

## Comprobaciones rápidas

```powershell
# Estado actual
Get-Content D:\FlyOperatorLab\work\learning\status.json -Raw | ConvertFrom-Json |
  Select-Object state,reason,stage,condition,phase,brain_mode

# Bloques de validación y actualizaciones de PPO de la campaña en curso
Get-Content "D:\FlyOperatorLab\work\learning\campaigns\v4.1-night1\blocks.jsonl" -Tail 3
Get-Content "D:\FlyOperatorLab\work\learning\campaigns\v4.1-night1\ppo_updates.jsonl" -Tail 3
```

En el panel: pestaña **Circuito** para el flujo sensorimotor, **Decisión** para la última
decisión con sus contribuciones, **Actividad** para las series.

## Reglas de la casa

- Una sola campaña a la vez: el bloqueo del trabajador impide dos controladores.
- Cambiar `config/learning.json` (PPO, acciones, máscaras, etapas) cambia el **hash semántico**:
  hay que usar un `experiment_id` nuevo, nunca reanudar una campaña vieja con otro protocolo.
- El entrenamiento se pausa solo si baja la RAM disponible de 1.0 GB (reanuda con 1.5 GB) o si
  los registros superan la cuota de 8 GB.
- El examen final del lunes usa el conjunto `test`, que no se toca en ninguna otra parte.
