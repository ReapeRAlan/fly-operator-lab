# Fly Operator v3.2 — operación y método científico

Fly Operator conserva **166,700 neuronas clasificadas y 25,582,938 conexiones dirigidas** de MaleCNS v1.0. La anatomía publicada permanece fija; la dinámica LIF, la interfaz sensorial, el catálogo de acciones y el aprendizaje son adaptaciones de ingeniería identificadas como tales.

La campaña activa es `pilot-v3.2-curriculum`. `pilot-v3.1-curriculum` se detuvo con checkpoint y queda como control histórico. Su meta es medir si un operador aprende habilidades y misiones. Una integración técnica correcta o un éxito con instructor no demuestra aprendizaje autónomo.

## Cambios v3.2 (16 de septiembre de 2026)

Evidencia completa en [outputs/INFORME_APRENDIZAJE_V32.md](outputs/INFORME_APRENDIZAJE_V32.md).

- **Rendimiento:** la elegibilidad solo se registra en aristas que entran en las 1,314 descendentes, las únicas que acredita la regla causal. Es idéntica bit a bit en el grafo real y el tick plástico es ~13× más rápido (`outputs/motor_cone_equivalence.json`).
- **Interfaz sensorial v3.2** (`data/learning/sensory_ports_v32.json`, generado con `scripts/build_sensory_schema_v32.py`): los mismos 98 canales y compuertas. La dirección y la proximidad del objetivo reciben las rutas anatómicas más fuertes, los sectores usan afinación von Mises y los estados lentos tienen ganancia baja. En la sonda de lazo abierto, las DN → acción pasan de 78.8 % a 99.1 % a ±45° (`scripts/probe_decodability.py`).
- **Plasticidad `causal_motor_rstdp_v2`:** el cambio de ganancia respeta el signo presináptico. La tasa se calibra con `scripts/calibrate_motor_plasticity.py`.
- **Imitación:** se conserva el procedimiento v3.1, que ya alcanza el techo lineal. El optimizador alternativo se evaluó offline, generalizó peor y quedó desactivado.
- **Operación:**
  - el plazo del piloto se guarda en `schedule.json` y un reinicio no lo extiende;
  - la RAM tiene histéresis (pausa < 1.0 GB, reanuda ≥ 1.5 GB; límites editables en caliente en `config/learning.json`);
  - las capturas antiguas se movieron a `archive/frames` (`scripts/archive_frames.py --restore` las devuelve).

Para iniciar una ventana nueva, por ejemplo hasta las 7:00:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\FlyOperatorLab\scripts\start_learning.ps1 -Hours 12 -ResetDeadline
```

Sin `-ResetDeadline`, un reinicio continúa con el plazo ya guardado. Si ese plazo venció, el entrenador termina de inmediato.

`start_learning.ps1` también inicia `scripts/supervise_learning.ps1`. Si el entrenador muere antes del plazo sin un paro intencional, el supervisor lo relanza desde el último checkpoint. Deja el registro en `work/learning/supervisor.log` y copia los logs del fallo como `crash_*`. "Guardar y detener piloto" en el panel termina también la supervisión.

## Abrir y leer el panel

Abre **http://127.0.0.1:8766** o `ABRIR OBSERVATORIO.url`.

- **Circuito (inicio):** flujo en vivo desde las entradas del juego hasta la acción, pasando por compuertas, familias de la red y neuronas descendentes. Incluye actividad por familia y la matriz de conectividad de las 25.6 M conexiones agrupadas en 8 familias (sinapsis o balance excitación/inhibición).
- **Aprendizaje:** separa enseñanza, práctica autónoma y validación. Muestra esperas libres, esperas mecánicas, probabilidad de esperar, entropía, curvas por condición y resultados reservados.
- **Percepción:** muestra salud, arma, munición, inventario, rehenes, civiles, aliados, enemigos visibles, puertas, objetivo, tiempo restante, mapa X/Z e historial persistente. El bloque “Qué recibió la red al decidir” usa la instantánea previa a la orden.
- **Actividad:** resume impulsos y neuronas activas de las 166,700 neuronas simuladas.
- **Conectividad:** grafo navegable de la neurona elegida (8 entradas y 8 salidas más fuertes, signo, familia y actividad actual) y la composición de sus entradas y salidas. Permite buscar `bodyId`, tipo o región y consultar anatomía, neurotransmisor predicho, eficacia simulada y trazas.
- **Decisión:** brújula con la probabilidad de cada dirección frente a la dirección del objetivo (última decisión libre), máscara mecánica, probabilidades y contribuciones de las 1,314 neuronas descendentes.

Desde otro dispositivo de la red Tailscale, el panel está en `http://<IP-de-Tailscale>:8766/` (`tailscale serve --tcp 8766`). Ahí es de solo lectura: pausar y detener requieren la PC del laboratorio.

El mapa es una reconstrucción de datos permitidos del motor. La red no recibe una captura de pantalla ni información de enemigos ocultos.

## Iniciar o continuar 24 horas

Conecta la alimentación y mantén la sesión de Windows abierta. La configuración actual admite **un solo monitor** y conserva el juego dentro de la pantalla disponible. Ejecuta:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\FlyOperatorLab\scripts\start_learning.ps1 -Hours 24
```

También puedes abrir `INICIAR APRENDIZAJE.cmd`. El lanzador:

1. verifica el hash del ejecutable y su PDB;
2. abre el juego con el perfil aislado, si hace falta;
3. respeta la política de pantalla de `config/runtime.json` (ahora `single_monitor`);
4. inicia una sola instancia del entrenador v3;
5. sirve el panel local.

**Pausar / Reanudar** se aplica en un límite seguro de paso. **Guardar y detener piloto** escribe un checkpoint y termina el entrenador. El juego puede mostrar *Planning Mode* mientras la red calcula; el reloj del motor sí avanza cuando el puente ejecuta el siguiente paso de 50 ms.

## Cambios que corrigen la campaña v2

### Percepción v3 comprobada

Hay **98 canales × 16 puertas neuronales = 1,568 puertos únicos**. Entre ellos se incluye la postura física confirmada por el motor (`crouched`).

- Los canales espaciales usan neuronas `visual_projection` excitatorias, ordenadas de forma reproducible por conectividad anatómica directa y a dos saltos hacia neuronas descendentes.
- Estado propio y tarea usan puertas `vnc_sensory` y `cb_sensory` excitatorias.
- Rehenes y civiles tienen canales explícitos, además de la señal general `friend_*`.
- Dirección y distancia del objetivo, tiempo restante y acción nativa en curso están codificados de forma separada.

La prueba pareada [sensory_transfer_v3.json](outputs/sensory_transfer_v3.json) activó cada sector con el mismo ruido de control en las semillas 7, 19 y 43. Los ocho sectores del objetivo produjeron ocho firmas distintas; cada sector cambió entre 82 y 173 neuronas descendentes por semilla. Enemigos, rehenes y 14 canales de estado propio —incluida la postura— también transfirieron señal, sin saturación de red en la ventana de 50 ms probada.

### Decisiones por macroacción

Una orden nativa puede tardar varios pasos. Esos pasos ahora se clasifican como transiciones mecánicas:

- no se guardan como etiquetas cognitivas `wait`;
- no entran como decisiones separadas a PPO;
- sus recompensas se acumulan sobre la acción que inició la orden;
- el descuento usa `gamma ** pasos_de_macroacción`;
- una nueva acción queda bloqueada mientras el recibo anterior siga `in_progress`, salvo dos intervenciones contextuales: `stop` cerca del destino y `cancel` durante una brecha;
- esas intervenciones terminan la macroacción anterior y reciben crédito causal propio.

Esto elimina el sesgo v2, donde 88.8–94.3 % de los ejemplos eran esperas y muchas solo representaban una orden pendiente.

### Episodios completos y DAgger por habilidad

PPO se actualiza al terminar un episodio natural. Ya no reinicia el mundo cada 256 pasos. Cada habilidad recibe enseñanza guiada y DAgger propios; DAgger combina la política con correcciones ocasionales del instructor y conserva solo decisiones reales. Las máscaras curriculares habilitan las familias necesarias para la habilidad actual.

Las políticas que repiten espera libre o no mejoran la distancia durante 32 decisiones se cortan con un motivo explícito. Esto limita trayectorias circulares sin confundirlas con éxitos o cierres técnicos.

### Plasticidad causal motora

La condición interna usa `causal_motor_rstdp_v1`:

1. congela una instantánea de elegibilidad **antes** de ejecutar la acción;
2. conserva solo aristas anatómicas activas que terminan en las 1,314 neuronas descendentes;
3. calcula crédito por descendente mediante el gradiente `d log pi(acción) / d DN` del actor;
4. aplica el error temporal al terminar la macroacción;
5. mantiene identidad, dirección, cantidad anatómica y signo de todas las conexiones.

El grafo completo sigue simulándose. Solo la actualización aprendida se restringe al cono motor causal. La R-STDP global v2 queda archivada como control histórico.

## Programa experimental

La secuencia es: movimiento y detención, postura, cancelación, orientación, cambio de arma, disparo, recarga, puertas, brecha, granadas, eliminación, rescate, desactivación y equipamiento.

Por semilla se crea un punto común y se comparan cuatro condiciones:

| Condición | Qué puede cambiar |
|---|---|
| Referencia congelada | Ningún parámetro que determine acciones |
| Adaptador | Actor y evaluador mediante PPO |
| Plasticidad interna | Eficacias del cono motor causal y evaluador; actor fijo |
| Combinada | Alterna episodios de adaptador y plasticidad interna |

La promoción exige al menos 80 % en validación para `adapter`, `internal` y `combined`, en las tres semillas, y retención de habilidades anteriores. La referencia congelada se reporta, pero no bloquea la promoción. La aceptación final conserva 30 episodios de validación por semilla; `test` permanece reservado.

Las evaluaciones cargan una copia del checkpoint y restauran cerebro, actor, filtros, optimizador, contadores y generadores aleatorios al terminar. No contaminan el estado que continúa entrenando.

Los pasos y episodios de una validación se mantienen en memoria hasta completar el bloque. Entonces se publican juntos en SQLite con un identificador de transacción. La pestaña de Percepción sigue mostrando una vista previa en vivo marcada como no comprometida. Si la laptop se apaga o se detiene el piloto a mitad del bloque, esas filas no entran en las tasas del panel. Las filas parciales anteriores a esta corrección se conservaron, fuera de las métricas activas, en `work/learning/archives/interrupted_evaluation_pre_transaction_v3.json.gz`.

## Checkpoints y trazabilidad

La campaña v3 vive en:

```text
work/learning/campaigns/pilot-v3.1-curriculum/
├── schedule.json
└── checkpoints/
    ├── common_7/current.json
    ├── adapter_7/current.json
    └── ...
```

Cada generación conserva:

- `brain.npz`: voltajes, estado sináptico, retardos, trazas, elegibilidades, eficacias, silenciamientos y RNG neuronal;
- `policy.pt`: actor, evaluador, optimizador, filtros descendentes y RNG de Torch/NumPy/Python;
- `examples.npz`: decisiones de enseñanza, máscaras y habilidad, con hash;
- `manifest.json`: hashes, semilla, condición, contadores y versión.

El manifiesto también fija un hash semántico de dinámica, recompensa, currículo, escenarios, capacidades, equipo, vocabulario de acciones y esquema sensorial. Los parámetros operativos de CPU/RAM/duración pueden ajustarse sin invalidar el aprendizaje. Un checkpoint con valores no finitos no reemplaza el último puntero válido.

El contrato de capacidades v2 compara las acciones validadas y su significado, y conserva la procedencia completa en archivos separados. Recompilar un puente equivalente por un ajuste de pantalla no invalida el aprendizaje; agregar, retirar o cambiar una capacidad sí lo invalida. La migración única del checkpoint anterior quedó registrada en `outputs/checkpoint_semantic_migration_v2.json` sin modificar cerebro, política, ejemplos, actor ni grafo.

`work/learning/experiments.sqlite` conserva pasos y episodios. `status.json` y `latest.json` alimentan el panel. El heap de Door Kickers 2 no se serializa: después de reiniciar se conserva el aprendizaje y se reinicia la misión interrumpida.

## Recursos y pausas

Perfil actual:

- afinidad lógica `[2, 4, 6, 8]`;
- PyTorch con 2 hilos;
- prioridad `BELOW_NORMAL`;
- descanso mínimo de 0.05 s por paso;
- juego limitado a 20 fps;
- reserva operativa de 1.5 GB de RAM disponible para esta sesión nocturna;
- límite de 4 GB para el proceso del entrenador;
- 8 GB libres en disco y cuota de artefactos de 8 GB.

El integrador neuronal es secuencial. Darle más núcleos no multiplica su velocidad. Si la RAM o el disco incumplen el límite, el entrenador guarda y pausa; se reanuda automáticamente cuando el recurso vuelve. La pantalla se mantiene despierta mientras entrena. Si después se configura `secondary_monitor`, la ausencia de esa pantalla volverá a ser una causa de pausa.

## Validaciones actuales

- Instructor en motor nativo: **42/42** combinaciones de 14 habilidades × 3 semillas, en [teacher_validation_v3.json](outputs/teacher_validation_v3.json).
- Transferencia sensorial v3: **aprobada**, en [sensory_transfer_v3.json](outputs/sensory_transfer_v3.json).
- Suite local: contratos neuronales, archivo sin pérdida, percepción, recursos, campañas, restauración post-evaluación, descuento de macroacciones y puerta de promoción.
- Parada/reanudación real: actor, contadores y ejemplos restaurados por hash antes de continuar.

Estas pruebas demuestran que la infraestructura funciona. La competencia autónoma y la mejora frente a la referencia congelada siguen pendientes de resultados reservados.

## Límites científicos

- MaleCNS aporta anatomía y predicciones; no aporta los parámetros fisiológicos completos usados aquí.
- Los conceptos “arma”, “rehén” y “objetivo” son asignaciones de ingeniería.
- Las contribuciones al logit son matemáticas; no son explicaciones biológicas.
- Una misión ganada con instructor no cuenta como aprendizaje.
- No se afirmará que el sistema resuelve misiones hasta superar el protocolo reservado en las tres semillas.

Informe v3: [outputs/INFORME_APRENDIZAJE_V3.md](outputs/INFORME_APRENDIZAJE_V3.md). La campaña v2 y su informe permanecen como control histórico.
