# Fly Operator v2: aprendizaje y observación científica

**Resultado:** el mecanismo de aprendizaje y el panel funcionan en pruebas locales con la red clasificada completa. El piloto está preparado para ejecutar enseñanza y práctica reanudable. La mejora sostenida en misiones reservadas todavía no está demostrada.

Informe generado: 2026-09-15T19:18:30. Estado observado: **running**; fase **teaching**, semilla **43**. El estado cambia durante el piloto; consultar [panel local](http://127.0.0.1:8766).

## 1. Lo implementado y lo comprobado

| Componente | Evidencia y alcance |
|---|---|
| Red completa clasificada | 166,700 nodos, 25,582,938 aristas dirigidas; sin poda adicional. Hash de grafo: `7f77d769d2fcba02f0581da8f9b3e913dd7d7d2abf067ddc71686a5b7545b161`. |
| Plasticidad | R-STDP con eficacias separadas, signo conservado y límites 0.25–4. La prueba cambió 2255377 eficacias; el actor permaneció idéntico durante esa actualización. |
| Adaptador | Actor lineal sobre 1,314 neuronas descendentes × filtros de 100/500/2,000 ms; imitación y MaskablePPO modificaron sus pesos en una prueba dentro del juego. |
| Reanudación | Restauración exacta de actividad, cola de retardos, RNG neuronal, pesos, filtros y RNG de PyTorch. Optimizador incluido y restaurado. |
| Puente | 20 ciclos consecutivos de acciones nativas, sin duplicación de secuencias ni cierre. Estos ciclos usaron órdenes de prueba, no aprendizaje neuronal. |
| Control | 16 familias habilitadas: aim_target, cancel, crouch, defuse, door_breach, door_open, equip, fire, follow, loadout, move, reload, stop, throw, turn, wait. |
| Equipo | 848 plantillas XML indexadas, atributos completos, relaciones de compatibilidad y parámetros efectivos del inventario. Tres conjuntos de equipo probados. |
| Interfaz | Cinco vistas, búsqueda por bodyId/tipo/región anotada, eficacia por arista, trazas observadas, probabilidades, contribuciones al logit y exportaciones. |
| Video | Muestra de 0.9 segundos de cambio de arma por actor sin instructor; un solo caso, fuera de las estadísticas de aceptación. MP4 1920×1080/30. |

La ampliación **no cubre todavía todas las familias del motor**: `use`, `clear_obstacle`, `arrest`, `spy_camera` y `evacuate` permanecen deshabilitadas hasta verificar sus efectos. El catálogo XML completo no implica que cada objeto sea una acción entrenable. La cancelación validada corresponde a apertura con herramienta; los ciclos de arma conservan su disponibilidad nativa.

Pruebas: [red y checkpoint](learning_full_validation.json), [aprendizaje dentro del juego](training_validation_v2.json), [acciones](native_actions_v2.json), [equipo y cancelación](native_kits_v2.json), [20 ciclos del puente](stability_v2.json), [calibración](plasticity_calibration_v2.json), [intervenciones](causal_probe_v2.json).

## 2. Qué se modela

Se simulan todas las filas con superclass identificada, incluidas categorías provisionales. Los segmentos sin clasificación quedan en el grafo crudo original. Hay 516 registros Traced sin superclass fuera de este inventario; no se presentan como neuronas modeladas. Los 211,577 registros de anotaciones no equivalen al número de neuronas simuladas. El archivo de auditoría conserva además conexiones hacia segmentos fuera del inventario.

Datos anatómicos y predicciones de neurotransmisores proceden de [MaleCNS v1.0](https://male-cns.janelia.org/download/). La conectividad no proporciona por sí sola constantes de membrana, eficacia eléctrica, aprendizaje, visión del juego o conceptos de armas. Esas asignaciones se documentan como supuestos de ingeniería. El modelo LIF toma como referencia la implementación de [Shiu](https://github.com/philshiu/Drosophila_brain_model); su modelo original usa otro conectoma.

La codificación usa 74 canales × 16 neuronas sensoriales asignadas de forma fija: entidades visibles, puertas, objetivo anunciado, rayos contra geometría conocida, estado propio y etapa. El actor recibe únicamente 3,942 valores derivados de actividad descendente. No recibe coordenadas ni salud ni munición de enemigos ocultos. Las restricciones de acciones sí reflejan disponibilidad mecánica. La búsqueda de región usa `somaNeuromere`, cuando está anotada; no equivale a una auditoría completa de innervación por neuropilo.

## 3. Dónde se almacena el aprendizaje

- `work/learning/checkpoints/<condición>_<semilla>/current.json`: manifiesto de la generación válida y hashes SHA-256.
- `<generación>/brain.npz`: eficacias por conexión, voltajes, estado sináptico, periodo refractario, retardos pendientes, huellas, elegibilidad, filtros de silenciamiento y RNG neuronal.
- `<generación>/policy.pt`: actor, evaluador, optimizador, filtros temporales, RNG de PyTorch/NumPy/Python y contadores.
- `work/learning/examples_<semilla>.npz`: ejemplos guiados y etiquetas agregadas en estados visitados por el aprendiz.
- `work/learning/experiments.sqlite`: observaciones permitidas, decisiones, recompensas, eficacia actualizada, episodios y checkpoints.
- `work/learning/schedule.json`: semillas preparadas, condición, ronda, etapa y resultados de validación.

Los estados neuronales se pueden repetir exactamente fuera del juego. El heap del motor no se serializa: al reabrir el entrenador se conserva el aprendizaje y se reinicia el episodio interrumpido. Los hashes verifican integridad antes de deserializar un checkpoint local. Los datos anatómicos originales permanecen inmutables.

## 4. Protocolo de entrenamiento

Cada semilla (7, 19, 43) recibe 24 episodios guiados y 8 episodios con etiquetas del instructor en estados del aprendiz, siguiendo el principio de [DAgger](https://proceedings.mlr.press/v15/ross11a.html). Tras esa preparación se crea el punto de partida común para las cuatro condiciones. La imitación se actualiza después de cada episodio; los ejemplos previos se conservan.

| Condición | Parámetros que cambian en práctica |
|---|---|
| Congelada | Ninguno de los que determinan acciones. |
| Adaptador | Actor y evaluador mediante PPO; eficacia interna fija. |
| Interna | Eficacia interna mediante R-STDP y evaluador de recompensa; actor fijo. |
| Combinada | Alterna bloques de PPO y R-STDP; las representaciones internas permanecen fijas durante cada bloque PPO. |

La señal de modulación es el error temporal de un evaluador externo: recompensa + γV(siguiente) − V(actual), sin bootstrap en terminación real y con bootstrap cuando hay un corte externo. No se identifica esa señal con una medición de dopamina. Referencias: [R-STDP](https://www.izhikevich.org/publications/dastdp.htm), [MaskablePPO y su ausencia de política recurrente](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

Las rondas asignan 256 decisiones por condición/semilla. Cuando un bloque termina a mitad de episodio se registra `truncated`, motivo `condition_switch_budget`, conservando el bootstrap. Esto limita cada tramo a 12.8 segundos; para estudiar misiones largas debe ampliarse `ppo.n_steps` y recalibrarse el presupuesto. La comparación registra tiempo simulado y real por separado.

Las evaluaciones se programan cada 4,096 decisiones: 30 mapas reservados por semilla y por habilidad adquirida, sin instructor ni plasticidad. Las nueve combinaciones de condiciones que aprenden × semillas deben alcanzar 80% y conservar habilidades anteriores para promover la etapa común. La referencia congelada se evalúa y se reporta, pero su fracaso no bloquea indefinidamente la promoción. Uno de cada cuatro episodios de práctica revisita habilidades anteriores. Los mapas test se reservan para evaluación final mediante el script independiente.

La recompensa combina éxito, derrota, costo de tiempo y diferencias de potencial hacia el objetivo. No recompensa disparos ni recargas por sí mismos. La bonificación potencial usa γΦ(siguiente)−Φ(actual), con potencial terminal cero; las garantías teóricas dependen de los supuestos expuestos en [Ng, Harada y Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

## 5. Parámetros y calibración

Valores iniciales: paso 0.1 ms; decisión 50 ms; membrana 20 ms; sinapsis 5 ms; retardo 1.8 ms; reposo −52 mV; umbral −45 mV; refractario nominal 2.2 ms (separación mínima efectiva 2.3 ms con el orden de actualización); huellas 20 ms; elegibilidad 2 s; η=10⁻⁴; A+=1 y A−=1.05; eficacia ×0.25–4. PPO: tasa 3×10⁻⁴, γ=.999 y clip=.2. Las unidades y procedencia están en `/api/parameters`, `config/learning.json` y `src/learning_brain.py`.

Se probaron η de 10⁻⁵, 10⁻⁴ y 10⁻³ desde el mismo estado y ruido, con resultados finitos y eficacias dentro de límites. En esa ventana de 50 ms hubo 90.3% de neuronas silenciosas y 0.065% cerca de la frecuencia máxima del modelo. Esto caracteriza ese estímulo y esa ventana; no prueba calibración fisiológica ni estabilidad a largo plazo.

La última prueba completa ocupó 953.1 MiB en el proceso de simulación aislada. Las memorias auxiliares se guardan en archivos mapeados y los estados de elegibilidad vacíos tienen codificación compacta. Los hashes de impulsos antes/después de la optimización coincidieron. El entrenador añade PyTorch, ejemplos y registros; el panel muestra su disponibilidad real de RAM.

## 6. Intervenciones y video

La repetición desde el mismo estado produjo exactamente los mismos impulsos. Cambiar entradas modificó las probabilidades (distancia L1 0.006641); silenciar neuronas descendentes produjo L1 0.090782. Son intervenciones computacionales fuera del juego; no demuestran causalidad biológica ni éxito táctico. Las contribuciones lineales mostradas en el panel son asociaciones exactas con el logit, no explicaciones causales completas.

El botón Capturar ventana guarda checkpoint previo y los impulsos del siguiente paso. Las conexiones consultadas registran huellas pre/post y elegibilidad. La serie de voltaje comienza al consultar una neurona y se separa al reiniciar su estado. Se exportan SVG, PNG, CSV, parámetros y un paquete ZIP de evidencia.

[Clip técnico MP4](FlyOperator_investigacion_v2_1080p30.mp4): 0.9 segundos, cambio de arma completado en un mapa reservado con un checkpoint de integración que combina imitación, PPO y una actualización R-STDP. El rótulo lo identifica como prueba combinada. El archivo [de trazabilidad](research_clip_trace_v2.json) conserva la condición interna original `adapter` y añade la corrección de clasificación; esa etiqueta antigua no debe usarse para atribuir el resultado a adaptación exclusiva. Una muestra no satisface 30 episodios × tres semillas.

Los fotogramas nativos se capturan cada 50 ms (20 fps), se mantienen/retemporizan a 30 fps y se escalan con bandas a 1080p. Las esperas de cómputo no entran en el video. El manifiesto del clip documenta resolución de origen, tiempo y conversión.

## 7. Uso y continuidad

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\FlyOperatorLab\scripts\start_learning.ps1 -Hours 24
```

Abrir http://127.0.0.1:8766. Pausar/Reanudar opera al terminar el paso actual; Guardar y detener termina con checkpoint. El juego permanece en el monitor secundario y usa el perfil aislado. Una sola instancia controla la canalización. Prioridad baja, perfil quiet_v1 con un procesador lógico, prioridad IDLE del entrenador, descanso mínimo de 750 ms por paso y presentación limitada a 20 fps; CPU para aprendizaje, mínimo 3 GB RAM disponible y 8 GB disco libre. La cuota de 8 GB considera registros, capturas y entregables. Las pausas por recursos se reanudan automáticamente al recuperarlos; errores de transporte o numéricos detienen con motivo y checkpoint cuando es posible.

Un fallo de carga observado al arrancar antes de completar recursos se corrigió retrasando la primera solicitud hasta al menos diez segundos desde el lanzamiento. Las pruebas posteriores pasaron; un equipo mucho más lento puede requerir revisar esa guarda. La semilla solicitada controla geometría y RNG neuronal. El motor publica su propia `map_seed`, que se registra; no se afirma reproducción exacta de toda la física del juego entre reinicios.

Herramientas: `scripts/evaluate_checkpoint.py` para evaluar un checkpoint con el entrenador detenido; `scripts/causal_probe.py` para intervenciones fuera del juego; `scripts/calibrate_plasticity.py`; `scripts/export_evaluation_video.py`; `scripts/build_learning_report.py` para actualizar este informe. Entorno independiente: `.venv-learning` / Python 3.11, dependencias en `requirements-learning.lock.txt`.

## 8. Resultados comparativos observados

| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95% Wilson |
|---|---:|---|---|---:|---:|---|
| common_teaching | 7 | breach | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 7 | defuse | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 7 | door | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 7 | elimination | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 7 | grenade | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 7 | loadout | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 7 | move | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 7 | orient | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 7 | reload | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 7 | rescue | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 7 | shoot | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 7 | switch | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 19 | breach | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 19 | defuse | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 19 | door | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 19 | elimination | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 19 | grenade | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 19 | loadout | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 19 | move | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 19 | orient | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 19 | reload | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 19 | rescue | train | 2 | 2 | 34.2%–100.0% |
| common_teaching | 19 | shoot | train | 3 | 3 | 43.9%–100.0% |
| common_teaching | 19 | switch | train | 3 | 2 | 20.8%–93.9% |
| common_teaching | 43 | breach | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | defuse | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | door | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | elimination | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | grenade | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | loadout | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | move | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | orient | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | reload | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | rescue | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | shoot | train | 1 | 1 | 20.7%–100.0% |
| common_teaching | 43 | switch | train | 1 | 1 | 20.7%–100.0% |

Los éxitos de enseñanza usan instructor y no cuentan como aprendizaje demostrado. Los intervalos describen los episodios observados; las semillas se reportan por separado y no se simula un tamaño de muestra mayor. Las pérdidas protegidas, consumibles, acciones rechazadas, duración y costo de cálculo están en los registros por episodio y paso.

**Aceptación completa pendiente:** las cuatro condiciones × tres semillas, 30 episodios reservados por habilidad, retención y misiones finales. El piloto y el panel permiten medirlo; todavía no hay evidencia para afirmar que aprendió a resolver las misiones del juego.
