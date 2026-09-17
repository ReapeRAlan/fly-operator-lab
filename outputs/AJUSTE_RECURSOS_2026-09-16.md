# Ajuste de recursos — 16 de septiembre de 2026

El entrenador continúa con el perfil `performance_cpu_v1`, autorizado para el día en que la laptop no se usará para otros trabajos. Se conserva el vencimiento original del piloto: **16 de septiembre, 19:14 aproximadamente**, hora local.

## Ajuste aplicado

| Recurso | Anterior | Actual |
|---|---|---|
| Afinidad del entrenador | Procesador lógico 19, núcleo de eficiencia | Lógicos 2, 4, 6 y 8, cuatro núcleos de rendimiento |
| Hilos para operaciones PyTorch | 1 | 2 |
| Prioridad Windows | IDLE | BELOW_NORMAL |
| Descanso real entre decisiones | 750 ms | 50 ms |
| Reserva mínima de RAM | 3 GB | 2 GB |
| Límite de RAM del entrenador | 4 GB | 4 GB |
| Reserva de disco / cuota de registros | 8 GB / 8 GB | 8 GB / 8 GB |
| Presentación del juego | 20 fps | 20 fps |

La topología se leyó mediante `GetSystemCpuSetInformation`; quedó en `work/cpu_topology.json`. El integrador neuronal sigue siendo secuencial: permitir cuatro núcleos no implica que se usen cuatro simultáneamente. No se cambian las 166,700 neuronas, 25,582,938 conexiones, paso neuronal de 0.1 ms, decisión de 50 ms, parámetros de aprendizaje ni reglas de evaluación.

## Medición

Antes del reinicio se compararon muestras consecutivas de la misma fase de plasticidad interna y el mismo episodio:

- Antes: mediana **14.552 segundos** de cálculo neuronal, 16 decisiones.
- Después del cambio de afinidad y prioridad: mediana **5.731 segundos**, 21 decisiones; rango 5.571–6.078 segundos.
- Relación observada: **2.54 veces**. Las entradas y la actividad no fueron idénticas entre muestras; no es un ensayo controlado ni una garantía de aceleración para todas las fases.

Fuente: `outputs/resource_profile_benchmark.json`. El muestreo final de carga, después de reanudar con el nuevo límite de RAM, se guarda en `outputs/performance_cpu_v1_warm.json`.

En los **90 segundos de comprobación final**, el estado permaneció `running` y el contador avanzó de 640 a 655. CPU total media 19.63%, máxima 29.2%; RAM disponible mínima 2.715 GB; temperatura máxima observada de GPU 48 °C. Son mediciones cortas del equipo completo.

## Continuidad y verificaciones

Se guardaron y verificaron por SHA-256 los checkpoints de red y política antes de cada reinicio. El primer reinicio restauró 626 decisiones; el segundo restauró 629 y avanzó. Se conservaron eficacias aprendidas, actor y optimizador. La misión interrumpida se reinicia según el contrato existente: no se serializa la memoria completa del motor.

La reserva inicial de 3 GB produjo una pausa automática real durante la carga. Se ajustó a 2 GB, el mínimo del plan original, con la autorización de usar más recursos; luego se comprobó que volviera a avanzar. La prueba que incluye esa pausa se conserva en `outputs/performance_performance_cpu_v1.json`; no debe confundirse con el muestreo final.

Once pruebas de configuración de recursos y contratos pasaron. El registro de errores del trabajador estaba vacío en la revisión y la ventana del juego quedó dentro del monitor secundario. Evidencias: `outputs/performance_profile_validation.json` y `outputs/performance_profile_validation_first.json`. Configuración anterior: `work/learning_config_before_performance.json`.

Estos controles comprueban continuidad técnica. No demuestran aprendizaje de misiones ni estabilidad térmica durante todo el día. La temperatura de GPU no sustituye una medición de temperatura de CPU.
