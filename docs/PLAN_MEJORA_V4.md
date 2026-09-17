# Plan de mejora v4

**Estado:** propuesto el 17 de septiembre de 2026; no está implementado.
**Base:** resultados de la campaña `pilot-v3.2-curriculum` ([RESULTADOS.md](RESULTADOS.md)) e investigación externa ([INVESTIGACION_REFERENCIAS.md](INVESTIGACION_REFERENCIAS.md)).

## Por qué hace falta

La noche del 16 al 17 de septiembre (11.5 h, 96,255 ticks, 91 rondas) dejó tres hechos:

1. **La interfaz v3.2 resolvió `move` sin aprendizaje autónomo.** `frozen` (sin aprender) acertó 92/92 episodios por semilla, igual que `internal`. Falta demostrar que algo *aprende*.
2. **El adaptador PPO desaprende.** Su práctica cayó por cuartos de la noche (semilla 7: 23/23 → 10/23 → 0/23 → 6/23; semilla 19: 22/23 → 10/23 → 0/23 → 0/23). Sacó 0/5 en validación en las semillas 19 y 43 y su entropía bajó a 0.04–0.18.
3. **La puerta de promoción nunca abrió**, porque el adaptador falló. Aunque abriera, exigir 512 decisiones por condición y semilla haría que habilidades de 1–2 decisiones (agacharse, cambiar arma) tardaran días. Las etapas de armas quedaron fuera de alcance.

Los proyectos revisados coinciden: conectoma congelado con lectura entrenada, imitación → RL anclado, jerarquía de comandos, y aprendizaje interno por tipo celular con gradientes. Ninguno logró que la plasticidad por arista mejorara la conducta.

---

## Fase A — próxima noche (sin GPU)

### A1. PPO anclado al instructor, sin colapso
**Archivos:** [`src/learning_curriculum.py`](../src/learning_curriculum.py) (`train_adapter_episode`), [`src/learning_policy.py`](../src/learning_policy.py) (`make_model`), [`scripts/train_curriculum.py`](../scripts/train_curriculum.py), [`config/learning.json`](../config/learning.json).

1. **Rollouts grandes:** acumular transiciones de varios episodios hasta ≥ 1,024 decisiones cognitivas antes de actualizar, en vez de una actualización por episodio. Conservar el descuento `γ^k` por macroacción que ya calcula `train_adapter_episode`.
2. **Hiperparámetros:** `batch_size` 256, `n_epochs` 2–4, `clip_range` 0.1–0.2, `target_kl` 0.015, `normalize_advantage=False` (o σ con piso de 1), lr del actor 1e-5 y lr del crítico 1e-4, con grupos separados en el optimizador.
3. **Calentar el crítico:** en las primeras 2,048 decisiones de cada habilidad, actor congelado (lr 0) y solo pérdida de valor, hasta que la varianza explicada sea > 0.5. Después, rampa lineal del actor de 0 a 1e-5 durante 2,048 decisiones.
4. **Pérdida ancla (kickstarting con DAgger):** sobre los estados que visita el agente, añadir `β · CE_enmascarada(logits, acción_del_instructor)`. La etiqueta sale de `ExerciseTeacher.choose` en `advance()`, que hoy solo se llama al recolectar demostraciones. β empieza en 1.0 y se multiplica por 0.995 en cada actualización, con piso de 0.05. Alternativa: KL al actor clonado congelado, coeficiente 0.2.
5. **Reversión:** conservar el mejor checkpoint por éxito evaluado; si una evaluación cae más de 20 pp, volver a él.
6. **Evitar RL innecesario:** si la imitación sola ya aprueba la habilidad, promoverla y reservar PPO para las habilidades que no aprueben.

**Aceptación:** en una noche, el éxito de práctica del adaptador no cae más de 5 pp respecto a `frozen` en ninguna semilla, y su validación es ≥ 80 % en 3/3 semillas.

**Pruebas nuevas:**
- el rollout acumula episodios y no actualiza antes del umbral;
- el actor no cambia durante el calentamiento;
- la pérdida ancla baja la CE contra el instructor;
- `target_kl` corta las épocas.

### A2. Puerta de promoción por episodios
**Archivos:** `scripts/train_curriculum.py` (bloque que usa `evaluation_interval_steps` y el cierre de ronda), `config/learning.json`.

- Evaluar cada condición y semilla tras **N episodios de práctica** (p. ej. 12), no tras 512 decisiones.
- Promover cuando el **límite inferior de Wilson al 95 %** del éxito en validación sea ≥ 0.80: con 0 fallos alcanzan 16 episodios; con 1 fallo, 25.
- Evaluación secuencial en bloques de 8 episodios: se detiene en cuanto el límite inferior queda ≥ 0.80, o cuando el superior queda < 0.80.
- `frozen` se reporta pero no bloquea, como hoy.

**Aceptación:** `stance`, `cancel` y `switch` promovibles en < 1 h cada una si la imitación las resuelve.

### A3. Controles: ¿aporta algo el conectoma?
**Archivos:** [`src/offline_replay.py`](../src/offline_replay.py), [`scripts/probe_decodability.py`](../scripts/probe_decodability.py), un script nuevo `scripts/build_rewired_graph.py` y la opción `--graph` en `LearningBrain`.

1. **Grafo recableado con grado preservado:** barajar destinos manteniendo grados de entrada y salida, conteos y signos presinápticos, y guardarlo como un segundo `data/processed/*_rewired.npy`.
2. **Sonda offline** con tres lectores sobre los mismos episodios: DN del conectoma real, DN del recableado y los 98 canales sensoriales sin cerebro.
3. **Condición `sensor_only` en lazo cerrado:** actor lineal sobre los canales sensoriales, con la misma imitación.

**Aceptación:** una tabla en `outputs/controls_v4.json` y en el informe. Si el recableado o `sensor_only` igualan al conectoma real, se dice explícitamente.

### A4. Reclasificar la plasticidad interna
- `internal` queda como **control histórico**: la calibración de v3.2 fue negativa, y DOOMFLY y Frémaux 2010 predicen el fracaso sin un crítico por estímulo.
- Deja de ocupar tiempo de noche en el currículo principal: se corre en una sola semilla, o se sustituye por C1 cuando exista.

---

## Fase B — semana 1: llegar a las etapas de armas

### B1. Acciones jerárquicas
**Archivos:** [`src/learning_adapter.py`](../src/learning_adapter.py) (`ActionCatalog`), [`src/learning_env.py`](../src/learning_env.py) (`ExerciseTeacher` como fuente de habilidades), `scripts/train_curriculum.py` (`STAGE_ACTIONS`).

- Habilidades de bajo nivel **congeladas y validadas en el motor**:
  - `ir_al_objetivo` (usa `move_to` del instructor);
  - `apuntar_y_disparar(objetivo_visible)`;
  - `recargar`;
  - `abrir_o_brechar(puerta)`;
  - `lanzar(posición)`;
  - `cubrir_esquina`;
  - `detener` y `cancelar`.

  Cada una termina por un recibo nativo, como hoy (`action_receipt`).
- La lectura de las DN elige entre unas **10–15 habilidades** más dos impulsos continuos opcionales (sesgo de giro y velocidad), igual que el impulso descendente 2D de NeuroMechFly o las ~7 DN de Eon.
- Validar cada habilidad con `scripts/validate_teacher.py` (hoy 42/42 casos) antes de habilitarla.

**Aceptación:** `switch`, `shoot` y `reload` promovidas en ≤ 1 noche con la puerta de A2.

### B2. Aprender la lectura sin gastar cerebro
- Guardar el vector completo de **3,942 features** por decisión cognitiva (un npz comprimido por episodio en `work/learning/features/`), además de máscara, acción, recompensa y etiqueta del instructor.
- Entrenar el actor lineal con **BC ponderado por ventaja (AWAC)** sobre ese replay: muchas actualizaciones por rollout, sin volver a simular el cerebro.
- Reducir el ruido Poisson promediando 2–3 réplicas en las evaluaciones críticas.

### B3. Jump-Start RL para habilidades largas
- En `breach`, `rescue` y `defuse`, el instructor juega los primeros *h* pasos. *h* baja un 25 % cada vez que el límite inferior de Wilson supera 0.8.

### B4. Retención de habilidades previas
- Hoy uno de cada 4 episodios repite una etapa anterior en orden fijo. Cambiarlo por **Prioritized Level Replay**: puntaje por error de valor más obsolescencia (β = 0.1, ρ = 0.3).

---

## Fase C — investigación: aprender dentro del cerebro

### C1. Parámetros por tipo celular
- Parámetros:
  - por tipo celular MaleCNS: ganancia de salida, umbral y τ_m;
  - por par de tipos pre/post: escalar de sinapsis.
- Conteos, signos y cableado quedan fijos, como en flyvis (734 parámetros) y el circuito de dirección de la cabeza (57).
- Optimización, dos vías:
  1. **sustituto diferenciable** de tasas o graduado con dt = 20 ms en PyTorch, entrenado por BPTT sobre la tarea o por imitación de la lectura;
  2. **estrategias evolutivas** sobre unos cientos de ganancias por familia o tipo, evaluadas en episodios offline (B2).
- Regularización L2 hacia 1.0, ensamble de ≥ 5 semillas y control recableado (A3).

**Aceptación:** mejora sobre `frozen` en validación reservada, con IC que no se solape y la misma mejora ausente en el recableado.

### C2. Neuronas no espigadas
- ~57 % de las neuronas (incluido el lóbulo óptico) no producen espigas, y un LIF con retardos fijos no computa movimiento.
- Si se usan entradas visuales ricas, modelar esas poblaciones como graduadas.

### C3. Plasticidad solo con crítico
- Si se retoma: aristas KC→MBON con error de predicción por DAN (recompensa − predicción por estímulo), siguiendo Jürgensen 2024 y Bennett 2021.
- Prueba de condicionamiento preregistrada antes de usarla en el juego.

---

## Fase D — rendimiento

### D1. Integrador por eventos en GPU (RTX 4050)
- Kernel CUDA/Triton o CuPy del bucle completo de 500 pasos (0.1 ms), o GeNN.
- Referencias: flybrain a 2.4× tiempo real en una RTX 3060; fly-survivors a 1.2× en una RTX 4080S; hoy estamos en ~0.2× en CPU.
- Validación **estadística** contra el integrador CPU, porque no será bit a bit:
  - tasas por familia;
  - sonda DN ±45° de `scripts/probe_decodability.py`;
  - transferencia sensorial.
- Implica una campaña nueva.

### D2. Más muestras por hora
- Replay offline (B2) para la lectura.
- Evaluar varias instancias del juego (hoy el puente acepta una sesión local).

---

## Orden recomendado

| Noche | Contenido | Resultado esperado |
|---|---|---|
| 1 | A1 + A2 + A3 (offline de día) + A4 | Adaptador estable; `move` → `stance` → `cancel` promovidas; controles medidos |
| 2–3 | B1 + B2 | Etapas `switch`/`shoot`/`reload` |
| 4+ | B3, B4 | Etapas largas: puertas, brecha, rescate |
| Paralelo | C1 y D1 como investigación | Aprendizaje interno medible; ≥ 1× tiempo real |

## Riesgos y límites
- Cambiar PPO, la puerta de promoción o el catálogo de acciones altera el hash semántico ([`src/learning_checkpoint.py`](../src/learning_checkpoint.py)) y obliga a empezar la campaña v4 desde cero. v3.2 queda como control.
- Si A3 muestra que los canales sensoriales solos igualan a las DN, el reporte debe decirlo: sería evidencia de que, con esta interfaz, el conectoma no aporta a `move`.
- Los proyectos de aficionados citados reportan resultados en sus propios README, sin verificación independiente.
