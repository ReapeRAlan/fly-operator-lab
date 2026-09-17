# Cómo entrenan otros proyectos cerebros de mosca

Investigación del 17 de septiembre de 2026 para el plan v4. Pregunta: ¿cómo han entrenado otros proyectos cerebros o conectomas de mosca (y afines) en tareas complejas, y qué aplica a Fly Operator?

> **Advertencia sobre las fuentes.** Artículos revisados por pares, preprints y repositorios de aficionados aparecen mezclados. En los repositorios de aficionados, los resultados son lo que dicen sus propios README y nadie los ha verificado. Cada afirmación lleva su enlace.

## Conclusión

1. **Entre los trabajos revisados no encontramos una demostración comparable y controlada de que la plasticidad por arista, o el RL sobre un conectoma completo, mejore la conducta en un juego.**
   - DOOMFLY usa el mismo MaleCNS que nosotros y una regla dopaminérgica: según su README, falló sus validaciones.
   - nfly usa nuestra misma lectura de 1,314 DN. Con RL no llegó a lo que alcanzó una cabeza **MLP** clonada de una CNN (+8.0 en 3 episodios); la cabeza **lineal** clonada se quedó en −9.7.
2. **Lo que sí funciona:**
   - conectoma **congelado** con lectura o encoder/decoder entrenados;
   - **imitación primero** y después PPO con tasas bajas y crítico separado;
   - **jerarquía**: pocos comandos de alto nivel sobre controladores de bajo nivel congelados;
   - cuando se aprende "dentro" del conectoma, **parámetros compartidos por tipo celular entrenados con gradientes** sobre un modelo diferenciable.
3. **Nuestro colapso de PPO tiene causas probables** (más los errores de implementación de la [sección 6](#6-revisión-externa-17-sep)):
   - crítico sin entrenar;
   - normalización de la ventaja en lotes diminutos;
   - nada que ancle la política al clon;
   - tasa de aprendizaje alta.

## 1. Moscas con cuerpo simulado

### NeuroMechFly v2 / FlyGym (EPFL)
- Repo https://github.com/NeLy-EPFL/flygym · artículo Wang-Chen et al., *Nature Methods* 2024: https://www.nature.com/articles/s41592-024-02497-y
- **Controlador:** separado en "cerebro" y cordón nervioso. El cerebro manda un **impulso descendente 2D** (uno por lado) a seis osciladores de patas (CPG); el control híbrido CPG + correcciones sensoriales fue el más robusto.
- **Navegación visual y olfativa:**
  - módulo visual (CNN) entrenado **supervisado**;
  - solo un MLP pequeño de decisión entrenado con **SAC**, que emite un sesgo de giro.
- **Conectoma en el lazo:** FlyVision (Lappalainen) preentrenado para seguir a otra mosca, con umbral sobre neuronas T → impulso descendente, sin entrenamiento adicional ([tutorial](https://nely-epfl.github.io/flygym-gymnasium/tutorials/advanced_vision.html)).

### flybody (Janelia / DeepMind)
- Repo https://github.com/TuragaLab/flybody · Vaxenburg et al., *Nature* 2025: https://www.nature.com/articles/s41586-025-09029-4
- Controladores MLP (sin conectoma), entrenados con **DMPO**, un MPO distribucional fuera de política con replay y restricciones KL ([MPO](https://arxiv.org/abs/1806.06920)).
- Recompensa por imitación de captura de movimiento, del orden de 10^8–10^9 pasos.
- Vuelo guiado por visión: **controlador de bajo nivel congelado**; solo se entrenan una CNN y un MLP de alto nivel que emite comandos de dirección de baja dimensión.

### FlyGM — conectoma completo como política (2026)
- https://arxiv.org/abs/2602.17997
- Conectoma FlyWire como red de paso de mensajes con la matriz de pesos firmados **congelada**.
- **Lo que se entrena no es solo la lectura:**
  - un descriptor intrínseco por neurona (η_v);
  - una función de actualización MLP compartida, condicionada por esos descriptores;
  - encoder, decoder y compuertas aferentes.
- Entrenamiento: **imitación** (KL al experto más un MSE que se atenúa) **→ PPO**, con lr 1e-4 (caminar) y 1e-5 (vuelo) y `ReduceLROnPlateau`.
- Error de giro: 8.29° con el conectoma, 13.55° recableado con grados preservados y 125.36° con grafo Erdős-Rényi.
- **Su línea base "SNN" no es un simulador LIF como el nuestro:** es la misma arquitectura con activaciones LIF (τ = 2.0) y gradiente sustituto. Nunca logró una marcha estable (retorno medio 34.64 contra ≥ 334). No dice nada general sobre modelos LIF con conectoma.

### Eon Systems (2026)
- https://eon.systems/updates/embodied-brain-emulation
- LIF de Shiu (~140 K neuronas) acoplado a NeuroMechFly cada 15 ms. Unas 7 DN (DNa01/02 giro, oDN1 velocidad, acicalamiento, alimentación, escape) **disparan controladores preentrenados por imitación**.
- Sin aprendizaje ni plasticidad; ellos mismos llaman a la visión "algo decorativa".
- Críticas:
  - Carboncopies: https://carboncopies.org/Blog/Posts/FruitFlyNotUploaded/Post/
  - Mineault: https://www.neuroai.science/p/are-flies-playing-beat-saber (sugiere limitar el aprendizaje al cuerpo fungiforme).

## 2. Proyectos que usan MaleCNS (nuestro conectoma)

Lista comunitaria: https://github.com/cobanov/awesome-fly · https://github.com/PerturbationAI/awesome-fly-brain-agents

| Proyecto | Cerebro y lectura | Entrenamiento | Resultado reportado |
|---|---|---|---|
| **nfly** https://github.com/zhengxuyu/nfly | MaleCNS (166,700), encoder de retina hexagonal, **1,314 DN → cabeza** | PPO / RLlib APPO sobre ganancias por arista, sesgos, τ y cabeza | Pong: RL −20.5 tras hasta 916 K pasos; desde una cabeza MLP clonada, −14 a −16 tras 200 K. Cabeza **MLP** clonada de una CNN: +8.0 (episodios 19, 20 y −15); cabeza **lineal** clonada: −9.7. "RL has not yet found the head that supervision finds in 6,000 steps." |
| **DOOMFLY** https://github.com/nftechie/doomfly | Mismo grafo (25,582,938 aristas); lectura manual DNp20 (giro), DNpe017 (avanzar/disparar) | Plasticidad dopaminérgica en 4,184 aristas KC→MBON11 | v6 "failed its visual, conditioning and survival validation gates" |
| **fly-craftax** https://github.com/liuzihe02/fly-craftax | Cerebro congelado, **lectura lineal sobre DN** y cabeza de valor lineal | PPO: 8 entornos × 64 pasos (512 por actualización), 4 épocas | Sin ancla BC/KL |
| **FlyDoom** https://github.com/eganeganegan/flydoom | RNN de tasas sobre MaleCNS, con modos solo-lectura / pesos fijos / pesos entrenables | PPO 3–5 M pasos; controles con grafo aleatorio, recableado, MLP, GRU y LSTM | Sin resultados cuantitativos publicados |
| **FLYT3** https://github.com/seanphan/flyt3 | LIF en GPU con umbrales homeostáticos; 512 DN + 512 motoneuronas → lineal | REINFORCE con línea base de valor | — |
| **Fly.exe** https://github.com/Ibtisam-Mohammad/Fly.exe | MaleCNS LIF en GPU (GeNN) → 2 tasas DN → controlador de giro publicado | Sin entrenamiento | 0.015× tiempo real |
| **fruitfly-lab** https://github.com/webergithub/fruitfly-lab | Rumbo del objetivo como Poisson a LC10a; giro = DNa02 R−L + AOTU019 R−L | Sin entrenamiento, una calibración | Doom: 10.2 bajas/episodio contra 0.7 con cableado barajado |
| **FLM** https://github.com/nftechie/flm | MaleCNS congelado (red tanh) + lectura de 278 K parámetros | Solo lectura | Un control del mismo tamaño sin anatomía de mosca rindió **ligeramente mejor** |
| **flyhard** https://github.com/MarkUnthank/flyhard | Interfaces diseñadas a mano; pata delantera gira un volante | No documentado | 100/100 objetivos tras 600 actualizaciones (una semilla) |

## 3. Entrenar redes que respetan el conectoma

- **flyvis** (Lappalainen et al., *Nature* 2024)
  - Enlaces: https://github.com/TuragaLab/flyvis · https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/
  - Conteos y signos fijos; el modelo de la red tiene **734 parámetros libres** para 45,669 neuronas: 65 potenciales de reposo, 65 constantes de tiempo y 604 escalares por par de tipos celulares.
  - Aparte se entrena el **decodificador** convolucional hexagonal de flujo óptico (34 → 8 → 3 canales, núcleo 5): **7,427 parámetros**, calculados a partir del código de `DecoderGAVP` porque el artículo no da la cifra. El decodificador tiene 10 veces más parámetros que la red.
  - Neuronas graduadas (sin espigas), entrenadas con **BPTT** a dt = 20 ms sobre flujo óptico.
  - Ensamble de 50 modelos (se analizan los 10 mejores). A mejor tarea, tuning más realista.
- **Circuito de dirección de la cabeza** (Duan, Dong, Fiete 2025)
  - Enlace: https://www.biorxiv.org/content/10.1101/2025.05.26.655406v1.full
  - De 193 K a **57 parámetros** por tipo celular.
  - El ligado global y el cableado barajado fallan.
- **Beiran & Litwin-Kumar** (*Nat Neurosci* 2025)
  - Enlace: https://www.nature.com/articles/s41593-025-02080-4
  - Conectividad + salida de la tarea no determinan la red: muchos parámetros quedan sin fijar.
  - Remedios: regularizar, usar ensambles o añadir registros de actividad.
- **BrainTrace** (*Nat Commun* 2026)
  - Enlaces: https://pmc.ncbi.nlm.nih.gov/articles/PMC12913608/ · https://github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model
  - Modelo de espigas de FlyWire ajustado a imágenes de calcio con gradiente sustituto en línea (pp-prop), usando 8.9 GB de memoria.
- **Reservorios con conectoma**
  - BPU (larva): https://arxiv.org/abs/2507.10951
  - ESA (mosca adulta como red de estado de eco): https://www.esa.int/gsp/ACT/projects/fly_connectome/
- **Cuerpo fungiforme y dopamina**
  - Jürgensen et al. 2024 (regla de tres factores más homeostasis; los DAN reciben un error de predicción): https://pmc.ncbi.nlm.nih.gov/articles/PMC10824792/
  - Bennett et al. 2021: https://ideas.repec.org/a/nat/natcom/v12y2021i1d10.1038_s41467-021-22592-4.html
  - Jiang & Litwin-Kumar 2021 (meta-aprendizaje con BPTT): https://pmc.ncbi.nlm.nih.gov/articles/PMC8354444/
- **R-STDP** (Frémaux, Sprekeler & Gerstner 2010)
  - Enlace: https://www.jneurosci.org/content/30/40/13326
  - Solo funciona si la señal es recompensa − predicción **por estímulo**, lo que exige un crítico; si no, domina el término no supervisado.
- **e-prop** (Bellec et al. 2020): trazas de elegibilidad más señales de aprendizaje, cerca de BPTT. https://www.nature.com/articles/s41467-020-17236-y
- **Ports a GPU del modelo completo**
  - flybrain: MaleCNS v1.0 a 2.4× tiempo real en una RTX 3060 (8.9× en lote fp16). https://github.com/annel0/flybrain
    - Entre sus límites, el autor anota que la vía del lóbulo óptico (95,501 de 166,700 neuronas en su modelo) señaliza con potenciales graduados y queda modelada con espigas.
    - Es una advertencia sobre **su** modelo, no verificada por nosotros. No la usamos como afirmación general.
  - fly-survivors (Triton): **FlyWire v783** (no MaleCNS) a 1.2× tiempo real en una RTX 4080 SUPER. https://github.com/arisson2001rojas-design/fly-survivors
    - Su README observa que en **su** red LIF de retardos fijos T4/T5 no llegan a disparar desde etapas previas, y por eso inyecta contraste en Tm1/Tm2/Tm4/Tm9.
    - Es una observación de ese modelo, no un resultado general sobre LIF.
  - Loihi 2: https://arxiv.org/html/2508.16792v1 · banco de pruebas: https://github.com/eonsystemspbc/fly-brain

## 4. Afinar con RL una política clonada sin colapso

- **PIRLNav** https://arxiv.org/html/2301.07302
  - Afinar un actor clonado con un crítico nuevo produce una "rapid drop in performance".
  - Receta:
    1. solo el crítico durante 8 M pasos, con el actor congelado;
    2. rampa de lr del actor de 0 a 1.5e-5;
    3. PPO con 2 épocas.
  - RL aporta menos cuanto mejor es el clon.
- **VPT** https://arxiv.org/pdf/2206.11795: KL a la política clonada congelada (coef. 0.2, ×0.995 por actualización); sin KL se pierden habilidades.
- **Wołczyk et al., ICML 2024** https://arxiv.org/html/2402.02868: *kickstarting* (KL al maestro en las trayectorias propias) y pérdida BC auxiliar evitan olvidar habilidades.
- **DAPG** https://github.com/aravindr93/hand_dapg: término de demostraciones λ0·λ1^k (λ0 = 1e-2, λ1 = 0.95).
- **AWAC** https://arxiv.org/abs/2006.09359 y **RLPD** (lotes 50/50 entre datos en línea y demostraciones).
- **Jump-Start RL** https://arxiv.org/abs/2204.02372: el guía juega los primeros *h* pasos y *h* se reduce con el éxito.
- **Normalización de la ventaja:** Wijmans et al. (https://arxiv.org/abs/2012.06117) la encontraron dañina con σ pequeñas en minilotes ruidosos.
- **SB3 / MaskablePPO:**
  - Por defecto: `n_steps` 2048, `batch_size` 64, 10 épocas, `target_kl` None, `normalize_advantage` True.
  - `target_kl` corta las épocas cuando el KL aproximado supera 1.5×.
  - Documentación: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html · https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- **Detalles de PPO:** https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ · Andrychowicz et al. https://arxiv.org/pdf/2006.05990
- **Currículos:**
  - Prioritized Level Replay: https://proceedings.mlr.press/v139/jiang21b/jiang21b.pdf
  - Teacher-Student Curriculum: https://arxiv.org/abs/1707.00183
  - Intervalo de Wilson para puertas por episodios: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval

## 5. Qué aplica a Fly Operator, por impacto esperado

| # | Lección | Evidencia propia | Evidencia externa |
|---|---|---|---|
| 1 | Anclar PPO al instructor, calentar el crítico, rollouts grandes, `target_kl`, lr ~1e-5 | Adaptador: 100 % → 0 %, entropía 0.04 | PIRLNav, VPT, Wołczyk, FlyGM |
| 2 | Puerta de promoción por episodios (Wilson), no por 512 decisiones | Habilidades de 1–2 decisiones tardarían días | Wilson, SPRT, PLR |
| 3 | Controles: grafo recableado y lector solo de sensores | `frozen` = 100 %: no sabemos si el conectoma aporta | FLM, FlyGM, FlyDoom, flyvis |
| 4 | Acciones jerárquicas (habilidades congeladas + pocos comandos) | 100 acciones planas; RL explora mal | NeuroMechFly, flybody, Eon |
| 5 | Dejar la R-STDP por arista como control, no como vía principal | Calibración negativa | DOOMFLY, Frémaux 2010 |
| 6 | Aprender dentro del cerebro con parámetros por tipo celular y gradientes | Plasticidad por arista inerte | flyvis, circuito de dirección de la cabeza, BrainTrace |
| 7 | Más muestras: replay offline de features; GPU solo si el perfil muestra que el cerebro domina | 0.116× tiempo real de extremo a extremo (96,255 ticks × 50 ms en 11.5 h) | flybody (10^8+ pasos), flybrain, fly-survivors |

## 6. Revisión externa (17-sep)

Una revisión externa del plan v4 señaló problemas. Estos se confirmaron leyendo el código de v3.2:

| Hallazgo | Dónde | Efecto |
|---|---|---|
| `aim_target i` indexa objetivos por **ID ordenado**, mientras el encoder representa **sectores** | `ActionCatalog.targets` en `src/learning_adapter.py` | Con las mismas entradas, la misma acción apunta a lugares distintos |
| La máscara habilita `stop` solo cerca del objetivo | `ActionCatalog.mask` | La máscara resuelve parte de la tarea en lugar del agente |
| `shoot` cuenta como éxito cualquier disparo aceptado | `skill_success` en `src/learning_env.py` | No mide si disparar correspondía |
| El kit de `loadout` sale de `seed % 3` | `ExerciseTeacher` | La etiqueta no depende de nada observable |
| El buffer PPO dura un episodio | `train_adapter_episode` en `src/learning_curriculum.py` | Lotes diminutos y ventajas mal normalizadas |
| `MaskablePPO.train()` → `_update_learning_rate` sobrescribe la tasa de **todos** los grupos | SB3 | No existen tasas separadas de actor y crítico |
| Imitación y PPO comparten el optimizador Adam | `imitate` y el modelo PPO | Los momentos de BC contaminan los primeros pasos de PPO |
| La lista `needed` fija a mano adapter, internal y combined | `scripts/train_curriculum.py` | La puerta no se puede configurar |
| `combined` recibe la mitad de actualizaciones PPO y sus episodios internos entrenan el crítico | `scripts/train_curriculum.py` | Hipótesis: su mejor retención frente a `adapter` sería un "calentamiento implícito", no un efecto de la plasticidad |

Otras correcciones de la revisión, aplicadas arriba:
- las cifras de nfly (MLP contra lineal);
- lo que entrena FlyGM y qué es su línea base SNN;
- el decodificador de flyvis;
- el conectoma de fly-survivors;
- el tono de las conclusiones generales.

Otras dos precisiones:
- `.gitignore` no limpia el historial: este repositorio se publicó desde cero con esas exclusiones ya aplicadas.
- Sin archivo `LICENSE`, el código es visible pero **no** es de código abierto (ver [THIRD_PARTY.md](../THIRD_PARTY.md)).

El plan concreto está en [PLAN_MEJORA_V4.md](PLAN_MEJORA_V4.md) (versión 4.1, con la fase A0 de auditoría).
