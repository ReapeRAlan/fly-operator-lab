# Percepción, ejecución real y carga de la laptop

Revisión local del 15 de septiembre de 2026. Panel: http://127.0.0.1:8766. El estado vivo del entrenador prevalece sobre este documento.

## Resultado

Percepción permite examinar salud, munición, arma seleccionada, inventario completo con parámetros efectivos, velocidad, orientación, órdenes pendientes, daño reciente, personas visibles, objetivos y objetos anunciados. El mapa reconstruye posiciones reales X/Z, paredes conocidas y el recorrido registrado. No es una captura ni una entrada visual de la red.

El historial es persistente y paginado; se filtra por condición y habilidad. Elegir un paso cambia la percepción, mapa, inventario y canales a ese registro. El botón **Volver al presente** recupera el último paso recibido. La exportación JSON incluye el registro completo, recorrido y esquema de neuronas receptoras. Consultar el pasado no cambia el entrenamiento.

Se probó un registro real de rescate con un rehén visible (registro 5084, episodio 152, paso 39). El tipo proviene del enum `eHumanType` del PDB: 2 = rehén. En la adaptación neuronal actual, los rehenes comparten los canales `friend_*` con otros amistosos; no se modificó ese esquema ni los pesos al ampliar el panel.

## Planificación y movimiento

Había dos fenómenos distintos:

1. La simulación permanece pausada entre decisiones. La etiqueta nativa **Planning Mode** durante esas pausas es compatible con ejecutar acciones por pasos.
2. En los escenarios con operador inicial fijo, el panel de selección de tropas permanecía abierto aunque el servidor ya había completado el despliegue. Se completó su transición con `GameClient::OnDeployFinished`, sobre el hilo del cliente y solo mientras su panel seguía abierto. Se espera la sincronización antes de iniciar un episodio.

Prueba controlada con la DLL actual: 25 pasos × 50 ms = **1,250 ms**, desplazamiento **1.905 m**, y coincidencia exacta del reloj de cada fotograma con el del servidor. Sin nuevas órdenes, posición y reloj permanecieron estables. Se inspeccionó el fotograma final: panel de tropas retirado y reloj visible en 00:01. Esto comprueba ejecución del motor; no demuestra aprendizaje autónomo.

Evidencia: [planning_validation.json](planning_validation.json), pseudocódigo en `work/decompiled/GameClient_OnDeployFinished.c`, código en `native/bridge.cpp` y prueba reproducible `scripts/validate_planning.py`. Se conserva el hash de la DLL y sesión en cada validación.

## Validación

- 21 pruebas Python aprobadas: contratos de aprendizaje, guardado, dinámica, historial y archivo sin pérdida. Dos advertencias de deprecación de bibliotecas, sin fallos.
- Tres kits de armas probados; agacharse y cancelar comprobados.
- 20 ciclos nativos consecutivos aprobados con la DLL final; `stability_v2.json` conserva las secuencias y resultados.
- Panel validado con Playwright local en 1536×1024 y 390×844: identidad, contenido, ausencia de errores/overlay, selección de histórico, paginación, filtros, escala del mapa, bodyId receptores, exportación JSON y regreso al presente. Sin desbordamiento horizontal móvil. [perception_qa.json](perception_qa.json).
- La ventana del juego quedó dentro del monitor secundario; [display_validation.json](display_validation.json).
- Historial aislado por experimento y sesión del juego. Los registros antiguos sin sesión no se unen en una trayectoria supuestamente continua.

## Carga y continuidad

El usuario confirmó que la laptop recuperó fluidez al cerrar juego, entrenador y panel. Pausar solamente el entrenador no bastó. Se aplicó `quiet_v1`: un procesador lógico con prioridad IDLE, descanso mínimo de 750 ms por decisión y presentación del juego limitada a 20 fps reales. No cambian las 166,700 neuronas, 25,582,938 conexiones, parámetros de aprendizaje ni el paso simulado de 50 ms.

En 45 muestras de funcionamiento después del arranque: CPU total media **10.72%**, máximo **23.4%**, RAM disponible mínima **3.515 GB**, GPU máxima **43 °C**. El entrenador avanzó de 247 a 279 pasos en su contador actual. Un muestreo de estado no obtuvo instantánea (`None`); no equivale a una detención y los registros posteriores siguieron avanzando. Un paso incluye cómputo, descanso y trabajo del motor; estos muestreos no son una comparación controlada de rendimiento. No demuestran que desapareciera la lentitud percibida.

En la última muestra, el entrenador ocupaba aproximadamente 0.83 GB residentes y el juego 0.77 GB. Esos valores no incluyen todos los procesos auxiliares, memoria compartida, GPU ni el resto de Windows. [performance_quiet_warm.json](performance_quiet_warm.json).

La reserva automática es de 3 GB de RAM y 8 GB libres en disco, con cuota de registros de 8 GB. El entrenamiento no graba video continuamente. **No se obtuvo temperatura de CPU**: Windows rechazó el acceso a la instancia del sensor AWCC. La temperatura de GPU y un ensayo corto no certifican estabilidad térmica de 24 horas.

El entrenador restauró 193 pasos y el hash del actor guardado `a4d710c816922309c470bc25479887a840ac590abd64952a52879dec3b5c5626`, y siguió produciendo registros. La revisión de SQLite fue `ok`; los hashes de red y política de los tres checkpoints comunes coincidieron, sin claves duplicadas dentro de una sesión. [night_audit.json](night_audit.json).

La auditoría de este momento encontró 70 episodios de enseñanza completos; no son 70 evaluaciones autónomas. La comparación de cuatro condiciones, tres semillas y validación reservada sigue pendiente. La sesión piloto se reanudó por hasta 24 horas, sujeta a las guardas y al estado visible del panel.

### Optimización posterior del guardado

El usuario confirmó que la laptop responde bien con el perfil reducido. Se mantuvieron esos límites y el experimento neuronal. Se eliminó una segunda compresión/escritura de los mismos ejemplos al terminar cada ejercicio guiado.

La compresión NPZ usa ahora DEFLATE nivel 1, sin pérdida. En tres muestras alternadas con afinidad a un procesador lógico y prioridad IDLE:

| Archivo real | Mediana anterior | Mediana nueva | Tamaño anterior → nuevo |
|---|---:|---:|---:|
| Estado neuronal | 0.919 s | 0.802 s | 1.25 → 1.73 MB |
| Ejemplos de enseñanza | 2.467 s | 1.057 s | 17.54 → 19.16 MB |

Todas las matrices recargadas coincidieron byte por byte, incluyendo forma y tipo. La compresión de ejemplos fue 2.33 veces más rápida en esta muestra; **esto no mide una aceleración equivalente del cálculo neuronal**. Además se evita la escritura duplicada. Se aceptó el incremento de tamaño por archivo dentro de la cuota vigente. [Datos del ensayo](checkpoint_io_benchmark.json).

Los ejemplos nuevos se guardan con hash y cantidad dentro de la misma generación que los pesos, antes de publicar el manifiesto. Los checkpoints antiguos siguen siendo legibles. Un fallo al restaurar no publica un checkpoint de un modelo incompleto. La prueba real guardó, cerró y reinició: restauró **554 pasos, 554 ejemplos y el mismo hash del actor**, y luego avanzó. [Evidencia](checkpoint_io_live_validation.json). La sesión final quedó entrenando por hasta 24 horas desde ese reinicio, con el juego en el monitor secundario.

## Uso

1. Abrir el panel y actualizar la página con **Ctrl+R**.
2. Entrar a **Percepción**. Desplegar un canal para ver su fórmula, estímulo en Hz y 16 neuronas receptoras.
3. En **Historial de acciones**, elegir un paso; filtrar **Rescate** para buscar personas observadas. Cero rehenes visibles no significa ausencia de rehenes ocultos.
4. Exportar el paso como JSON o usar las exportaciones generales del panel.

Las señales mostradas son posteriores a la orden del paso y alimentan la siguiente decisión. La máscara mecánica se registra al elegir la orden, antes de ejecutarla. Nombres y parámetros completos de armas son telemetría; no todos se codifican en la red. Estos límites se muestran también en la interfaz.
