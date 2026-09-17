# Prototipo v1 y referencia histórica (15-sep-2026)

> Documento histórico: describe el prototipo sin aprendizaje y la reproducción original. La portada vigente está en [../README.md](../README.md).

**Versión activa:** [guía de aprendizaje continuo y panel](../README_APRENDIZAJE.md). Abre **http://127.0.0.1:8766**. Para iniciar el piloto utiliza **INICIAR APRENDIZAJE.cmd**. El entrenamiento usa Python 3.11 en `.venv-learning`.

## Referencia histórica del prototipo v1/v2

El contenido siguiente describe la primera versión sin aprendizaje. No ejecutes sus controladores al mismo tiempo que el entrenador v3.1.

Prototipo local para Windows x64. Simula las **166,700 neuronas clasificadas** de MaleCNS v1.0 y las **25,582,938 conexiones dirigidas entre ellas**. Python obtiene la percepción del operador, simula actividad LIF y convierte actividad de neuronas descendentes en órdenes nativas del juego. No hay aprendizaje táctico.

El lanzador coloca la ventana en el **monitor secundario** sin activar esa ventana. El juego usa un perfil aislado en `work/profile`. El perfil de Steam y el ejecutable original no se modifican.

## Abrir y ejecutar

Desde PowerShell, con el juego del laboratorio abierto o cerrado:

```powershell
cd D:\FlyOperatorLab
python scripts/run_local.py --episodes 20 --seconds 3
```

Para grabar 20 episodios de tres segundos a 30 fotogramas por segundo:

```powershell
python scripts/run_local.py --episodes 20 --seconds 3 --record
```

También se puede usar **Iniciar Fly Operator.cmd**. Ejecuta los 20 episodios en segundo plano, escribe el progreso en `work/user_run.log` y deja el juego pausado al terminar. El nombre de cada ejecución incluye fecha y hora. No iniciar dos controladores a la vez.

Cada acción avanza 1–50 ms de juego; mientras Python calcula, el juego permanece pausado. Cerrar el controlador también lo deja pausado. La misión de prueba despliega un operador con equipo original y un enemigo armado. El operador puede morir; ganar no es un criterio de funcionamiento.

## Compilación comprobada

- Door Kickers 2 v1.12, Steam build 20072026, nativo x64.
- SHA-256 EXE: `7f873f34dfe37dfe40840140baf42352ea8f6b8e144d652a261a3dbddf83d747`.
- PDB: GUID `a2347bb6-eca6-424f-8b5c-a6431e79860c`, revisión 7.
- Python 3.9.13, MSVC 14.50.35717 / Visual Studio 18 Community.
- Ghidra 12.1.3 con PDB; descompilación dirigida en `work/decompiled`.

El lanzador rechaza cambios del EXE o del PDB. Una actualización del juego requiere repetir la ingeniería inversa; no basta con cambiar el hash.

## Reproducir desde los insumos públicos

Requiere Python x64 3.9 y las herramientas C++ indicadas. En este equipo todos los insumos y el DLL compilado ya están disponibles. La descarga de datos usa acceso público, **no requiere token de neuPrint**.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/acquire.py dataset
.venv\Scripts\python scripts/acquire.py toolchain
.venv\Scripts\python scripts/prepare_connectome.py
.venv\Scripts\python scripts/make_adapter.py
.venv\Scripts\python scripts/setup_profile.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_native.ps1
.venv\Scripts\python scripts/launch.py
.venv\Scripts\python scripts/validate_controls.py
.venv\Scripts\python scripts/validate_brain.py
.venv\Scripts\python scripts/audit_graph.py
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python scripts/run_experiment.py --episodes 20 --seconds 3 --interventions --name nueva_validacion
```

El script de compilación usa la instalación de Visual Studio presente en este equipo. Los scripts de inspección de binarios y los símbolos derivados están en `scripts/inspect_binary.py`, `scripts/native_profile.py`, `work/re` y `native/build_profile.h`.

Cierra el juego del laboratorio antes de recompilar el DLL, porque Windows mantiene ese archivo en uso mientras está cargado.

## Qué significa «completo» aquí

Se conserva todo el inventario con `superclass` publicada, incluidas clases provisionales `tbc`, neuronas sin conexiones y todas sus aristas de peso positivo. No se selecciona una región, no se limita a DNge104 y no se aplica un umbral adicional de peso. La descarga original incluye además segmentos sin clasificación neuronal y glía: esos registros se auditan y se conservan en bruto; no se convierten en neuronas LIF ficticias. **No se afirma simular los 151.86 millones de pares del grafo bruto de segmentos.**

El [cuaderno de los autores](https://github.com/flyconnectome/2025malecns/blob/main/supplemental_data/quantify-neuron-connections.ipynb) identifica las conexiones neuronales por las superclases de sus extremos. Ese cuaderno usa v0.9 y excluye `tbc`; este prototipo aplica el criterio a v1.0 e incluye también esas clases. `data/processed/audit.json` registra la cobertura exacta y las conexiones hacia segmentos no clasificados.

## Dinámica y adaptación

La anatomía procede de los [datos oficiales MaleCNS](https://male-cns.janelia.org/download/). La dinámica se inspira en el [modelo de Shiu](https://github.com/philshiu/Drosophila_brain_model), que fue desarrollado para FlyWire. Esta adaptación a MaleCNS no es una validación fisiológica.

- Paso neuronal 0.1 ms; reposo/reinicio −52 mV; umbral −45 mV.
- Constantes de decaimiento: membrana 20 ms y corriente sináptica 5 ms.
- Demora sináptica 1.8 ms; contador refractario de 22 pasos después de cada disparo.
- Peso dinámico = número entero de sinapsis × 0.275 × signo presináptico.
- GABA, glutamato e histamina se asumen inhibitorios; las demás etiquetas, incluidos neurotransmisores inciertos y moduladores, se aproximan como excitatorias. Esto es una simplificación explícita.
- Estímulos Poisson de 5 + 145 × intensidad Hz, con impulsos de 68.75 mV sobre los puertos sensoriales.
- Se usan 128 puertos por canal sensorial. **Elegir puertos no reduce la red simulada.**
- Las 1,314 neuronas descendentes se asignan por un hash fijo a siete grupos de lectura. Gana el grupo con mayor tasa media: mover, detener, girar a izquierda/derecha, apuntar, disparar o recargar. Los nombres de estas acciones son convenciones de ingeniería.
- El silencio neuronal produce `stop`. Ninguna regla sensorial selecciona directamente la acción.

`config/adapter.json` contiene los IDs biológicos y los índices exactos de cada puerto. `src/flybrain.py` contiene todos los parámetros, la integración y el generador pseudoaleatorio reproducible.

## Evidencia y límites

Consultar `outputs/INFORME.md`, los JSON de validación y las trazas por paso. El control se ejecuta en el hilo de simulación, mientras que la captura ocurre en el hilo de renderizado después de recibir el estado del servidor. Los fotogramas registran tiempo del servidor y del cliente.

La prueba de recarga usa una segunda disposición de la misma misión: el enemigo queda encerrado por paredes normales. Evita que la muerte del operador interrumpa la prueba; no modifica salud, munición ni cadencia. Los episodios neuronales usan la arena abierta.

El puente está limitado a una sesión local. No se validó cooperativo, campañas, otros operadores, otras versiones, aprendizaje, coordinación ni repetición nativa. El video procede de capturas directas de OpenGL, no de una repetición del juego.

Los datos son CC-BY según la página oficial. Se conservan las licencias de las herramientas en `third_party`; el juego original se necesita instalado.
