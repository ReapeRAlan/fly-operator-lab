# Fly Operator: resultado de implementación y validación

**15 de septiembre de 2026 — Door Kickers 2 v1.12 / MaleCNS v1.0**

## Dictamen

**El prototipo es funcional como experimento local con el inventario completo de 166,700 neuronas clasificadas de MaleCNS.** Se comprobó un ciclo real de percepción del juego, actividad LIF, lectura neuronal y control de un operador. Las conexiones permanecen fijas; no existe aprendizaje táctico. La cobertura precisa del inventario se detalla abajo.

Hay dos series de 20 episodios sin cierres, bloqueos ni secuencias duplicadas. La primera usó pasos de 50 ms; la segunda grabó a 30 fps. Los episodios tienen un horizonte de tres segundos y terminan antes si acaba la misión. Esta es una prueba corta de funcionamiento y estabilidad, no una certificación de campañas largas.

El juego queda **pausado en el monitor secundario**. El lanzador del laboratorio vuelve a colocarlo ahí en cada apertura.

## Entregables

- [Video de 60 segundos, 1080p/30 fps](Fly_Operator_MaleCNS_1080p30.mp4).
- [Código, parámetros y comandos de reproducción](../README.md).
- [Aceptación comprobada](acceptance.json), [control nativo](control_validation.json), [conectividad](graph_validation.json), [dinámica neuronal](brain_benchmark.json).
- [Serie de 20 episodios](live_validation/summary.json), [serie grabada](recorded_validation/summary.json) y sus archivos `trace.jsonl` con cada observación, actividad, orden y respuesta.
- Ingeniería inversa: `../work/decompiled`, `../work/re` y `../work/ghidra/DK2Fly.gpr`.

## Resultados medidos

| Requisito | Resultado |
|---|---|
| Mover, detener, orientar, apuntar, disparar y recargar | Comprobados en ejecución con rutinas nativas |
| Impedir disparo y recarga automáticos | Munición estable sin orden; cargador vacío durante 5 s sin recarga automática |
| Cadencia y recarga normales | Un disparo consume munición; se respeta el estado de enfriamiento; la recarga tarda y usa la cola nativa |
| Órdenes atrasadas | Rechazo de secuencia duplicada sin avance ni nuevo disparo |
| Desconexión | Tiempo y secuencia permanecen inmóviles después de cerrar Python |
| Red neuronal | 166,700 nodos; 25,582,938 pares; 124,177,617 sinapsis internas |
| Integridad | Hashes de descargas, sumas globales y contraste independiente de 40,952 pares de 256 neuronas |
| Reproducibilidad neuronal | Mismas entradas y semilla: mismos conteos por neurona; cinco pruebas numéricas aprobadas |
| Estabilidad | 20 + 20 episodios; 2,841 pasos verificados, más 159 de grabación complementaria |
| Video | 1,800 fotogramas decodificados sin errores; 60.000 s; 1920 × 1080; 30 fps |

Se desactivó `BrainPlayer::Think` únicamente para el operador elegido. El contador de intercepciones confirma su bloqueo en los pasos de simulación. No se sobrescribieron posiciones, vida o munición para producir los resultados de control. La primera prueba usa un enemigo visible y armado. Para comprobar recarga durante una espera larga, otra disposición encierra al enemigo con paredes originales; así la muerte del operador no interrumpe el ensayo.

### La red determina las acciones

Las tres condiciones iniciales usan la misma semilla neuronal. La intervención invierte las tasas de estimulación entre los puertos, sin cambiar la anatomía. En la desconexión se sigue simulando toda la red, pero sus órdenes se sustituyen explícitamente por detener.

| Condición, 3 s simulados | Disparos neuronales | Desplazamiento | Órdenes |
|---|---:|---:|---|
| Normales | 5,875,070 | 2.36 m | reload: 49, move: 9, stop: 2 |
| Alteradas | 5,907,633 | 3.18 m | reload: 19, move: 35, stop: 6 |
| Salidas desconectadas | 5,840,748 | 0.00 m | stop: 60 |

Las acciones propuestas por el decodificador coinciden con las ejecutadas en todas las filas normales y alteradas. Con salidas desconectadas, se registran disparos neuronales pero no desplazamiento. La prueba numérica complementaria usa exactamente el mismo estímulo para demostrar reproducibilidad, independientemente de las variaciones aleatorias propias del juego.

En la serie grabada hubo **3 eliminaciones del enemigo por órdenes del circuito**. No constituyen evidencia de aprendizaje: la arena tiene un blanco sencillo frente al operador y el adaptador de acciones es fijo.

## Alcance de MaleCNS completo

Los archivos oficiales contienen **211,577 registros de anotaciones** y **151,856,684 pares entre segmentos**. No todos son neuronas completas. El modelo conserva los **166,700 registros con `superclass`**, incluidas clases provisionales `tbc`, sin filtrar región, tipo, cantidad de sinapsis ni neuronas aisladas. Se incluyen todas las aristas entre esos registros.

Este criterio sigue la identificación por superclase del [cuaderno de los autores](https://github.com/flyconnectome/2025malecns/blob/main/supplemental_data/quantify-neuron-connections.ipynb). El cuaderno publicado usa v0.9 y excluye clases `tbc`; aquí se usa v1.0 y se conservan también esas clases. Los insumos y su licencia CC-BY están en la [descarga oficial](https://male-cns.janelia.org/download/).

Fuera del inventario clasificado quedan registros con estado Orphan (15,893), Glia (11,864), Unimportant (10,751), sin estado (3,470), Assign (1,832), Anchor (551) y **Traced (516)**. No se afirma que todos ellos sean glía o fragmentos. Las 117,436,340 aristas que cruzan hacia segmentos no clasificados y las 8,837,406 aristas entre otros segmentos se retienen en bruto y se contabilizan; **no participan en la simulación LIF de neuronas identificadas**. Esta distinción limita el significado de «completo»: no se simula cada segmento del volumen ni se resuelve su identidad biológica.

El archivo `../data/processed/audit.json` contiene la auditoría. `../data/raw/*.provenance.json` conserva URL, tamaño, SHA-256 y comprobación MD5 cuando el origen la proporciona.

## Dinámica: datos frente a supuestos

Son datos publicados los IDs, las anotaciones, la dirección de los pares, sus cantidades enteras de sinapsis y las predicciones de neurotransmisor. Son supuestos del prototipo las constantes LIF, el signo funcional asignado a cada neurotransmisor, la ganancia sináptica, la estimulación y la correspondencia con acciones humanas.

La implementación toma como referencia el [modelo de Shiu para FlyWire](https://github.com/philshiu/Drosophila_brain_model); esta adaptación a MaleCNS no ha sido validada fisiológicamente. El paso es 0.1 ms, con reposo/reinicio −52 mV, umbral −45 mV, constantes de 20 y 5 ms, demora de 1.8 ms y contador refractario de 22 pasos. El peso es 0.275 por sinapsis con el signo presináptico asumido. Los transmisores moduladores e inciertos reciben una aproximación simplificada, documentada en el README.

Se estimulan 128 puertos por cada uno de seis canales y se leen 1,314 neuronas descendentes distribuidas por hash entre siete acciones. **La selección de puertos no elimina otras neuronas de la simulación.** Los hashes de actividad de la traza resumen voltajes, corrientes y contadores refractarios; no son archivos completos de restauración del simulador.

## Ingeniería inversa y puente

Se inspeccionó el PE x64, se verificó la coincidencia EXE/PDB y se descompilaron **44 rutinas** con Ghidra. Es pseudocódigo reconstruido de funciones relevantes, no el código fuente original ni una descompilación integral de cada función del ejecutable.

| Área | Rutinas y aplicación |
|---|---|
| Decisiones | `BrainPlayer::Think`, `DoEngage`, `Human_TryShootingFirearm`; bloqueo del operador seleccionado |
| Acciones | `CmdAimInDirection`, `CmdFirearmShoot`, `ReplaceWaypoints`, `ProcessCmdFirearmReload` |
| Armas | `Firearm::ReadyAim`, `Fire`, `Reload`, `InitBullets`; verificación de estado y munición |
| Reloj | `GameServer::Update`, `UpdateGame`, `IsGamePaused`; paso virtual y banderas de pausa por cliente |
| Percepción | `OnVisStart`, `OnVisEntityInView`, `OnVisEnd`, `UpdateFOV`; espera del trabajo de visibilidad antes de observar |
| Sesión | Comandos nativos de carga y despliegue; reinicio verificado después de terminar una misión |
| Grabación | `SwapBuffers`/OpenGL; captura con tiempo del cliente y servidor coincidente |

Una canalización local de Windows transporta JSON con longitud prefijada. El servidor valida episodio y secuencia, y ejecuta órdenes en el hilo de simulación. El DLL usa MinHook y se carga antes de iniciar el ejecutable para aislar el perfil. El lanzador valida SHA-256 del EXE y PDB; el DLL vuelve a validar el EXE. Solo está admitida una sesión local.

La compilación comprobada es Steam build 20072026 / v1.12. EXE SHA-256: `7f873f34dfe37dfe40840140baf42352ea8f6b8e144d652a261a3dbddf83d747`. PDB GUID `a2347bb6-eca6-424f-8b5c-a6431e79860c`, revisión 7. DLL probado: `3cc923c2d43c70494382b095467f200eca0fe138cff4cdd73d80e18d8c3287e2`.

## Rendimiento y video

La primera serie simuló 60 s en **353.3 s reales**, incluidos reinicios: aproximadamente 5.89 veces el tiempo simulado. Registró **117,765,647 disparos neuronales**. El máximo RSS muestreado de Python fue **283.1 MiB**; no incluye el juego, herramientas o el resto del sistema. La simulación utilizó CPU en el i7-13650HX; no necesitó la RTX 4050.

La captura del minuto exportado necesitó **671.3 s reales**. El video reúne 21 episodios: 20 de la serie y uno complementario, porque tres misiones finalizaron antes del horizonte. Se eliminan las esperas de cálculo y de carga; los cambios de episodio están rotulados. El último paso dura 35 ms para ajustar la suma exacta a 60,000 ms.

Las capturas originales son de **1264 × 711 píxeles**, correspondientes a la ventana en el monitor secundario. El MP4 se amplió a 1920 × 1080; no es captura nativa de 1080p. Cada imagen se contrasta con ambos relojes internos. La telemetría del video identifica la acción y actividad del paso capturado. El indicador de planificación permanece visible porque el experimento mantiene la pausa entre pasos. La repetición nativa del juego no se usó para esta evidencia.

## Estado final y límites

- [Ubicación de ventana](display_validation.json): completamente dentro del monitor secundario, sin cambiar el monitor principal.
- [Perfil original](profile_validation.json): los 12 archivos comprobados coinciden con el respaldo. Las opciones se restauraron tras una prueba inicial anterior al aislamiento.
- [Estado final](final_state.json): juego pausado y canalización disponible para el próximo controlador.
- Sin validación de campañas largas, cooperativo, cambios de versión, elección de equipo, coordinación de escuadrón o aprendizaje.
- Se mantienen los archivos de ingeniería inversa y los datos en `D:\FlyOperatorLab`. El ejecutable y los recursos originales del juego son necesarios para reproducirlo.

**Conclusión:** se demostró viabilidad y funcionamiento del ciclo completo para el inventario neuronal clasificado de MaleCNS v1.0, dentro de la misión local de prueba. La validez biológica de la dinámica y la competencia táctica quedan abiertas.
