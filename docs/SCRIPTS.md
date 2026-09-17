# Inventario de scripts y módulos

Todos los scripts se ejecutan desde la raíz del proyecto. El entrenamiento v3 usa `.venv-learning\Scripts\python.exe` (Python 3.11); el prototipo v1, `.venv` (Python 3.9).

**Marcas:**
- **[juego]:** necesita el juego del laboratorio abierto con el puente.
- **[RE]:** análisis del binario del juego; requiere Ghidra y el PDB local.
- **[histórico]:** pertenece a v1/v2 y se conserva por trazabilidad.

## Adquisición y preparación
| Script | Propósito |
|---|---|
| `scripts/acquire.py dataset\|toolchain` | Descarga los insumos públicos con versión fija (conectoma MaleCNS; Ghidra, JDK, MinHook, nlohmann/json, stb y el modelo Shiu), verifica hashes y guarda la procedencia |
| `scripts/prepare_connectome.py` | Conserva todas las aristas publicadas entre las 166,700 neuronas clasificadas → `data/processed/` |
| `scripts/audit_graph.py` | Concilia de forma independiente filas de adyacencia muestreadas contra los IDs originales |
| `scripts/prepare_learning.py` | Genera el catálogo de equipo (desde los XML del juego instalado) y los escenarios de laboratorio deterministas; nunca edita archivos del juego |
| `scripts/setup_profile.py` | Crea el perfil aislado del juego en `work/profile` y respalda el original |
| `scripts/make_adapter.py` | [histórico] Mapeo artificial de entrada/salida del prototipo v1 |
| `scripts/build_sensory_schema_v32.py` | Esquema sensorial v3.2: asignación de compuertas por relevancia, afinación von Mises y ganancias; `--verify-v31` reproduce v3.1 |

## Puente nativo e ingeniería inversa
| Script | Propósito |
|---|---|
| `scripts/inspect_binary.py` | [RE] Inspección estática repetible del binario local autorizado |
| `scripts/research_v2.py` | [RE] Extrae layouts del PDB y objetivos de descompilación del protocolo v2 |
| `scripts/extra_targets.py` | [RE] Lista direcciones de funciones adicionales para Ghidra |
| `scripts/native_profile.py` | [RE] Genera `native/build_profile.h` (RVAs y hash) desde los símbolos |
| `scripts/build_native.ps1` | Compila `work/build/flybridge.dll` con MSVC, MinHook y json.hpp |
| `scripts/launch.py` | Lanza el juego con hash verificado en modo suspendido, inyecta el puente y lo reanuda |
| `scripts/verify_display.py` | Verifica en solo lectura la política de pantalla configurada |
| `scripts/keep_display_awake.py` | Mantiene despierta la pantalla durante corridas desatendidas |

## Validaciones con el juego [juego]
| Script | Propósito |
|---|---|
| `scripts/validate_controls.py` | Integración real de controles, sin observaciones sintéticas |
| `scripts/validate_planning.py` | Sincronía de tiempo, movimiento y cuadros del cliente |
| `scripts/validate_actions_v2.py` | Prueba cada acción nativa v2 con su criterio de efecto |
| `scripts/validate_kits_v2.py` | Prueba los tres kits de loadout |
| `scripts/validate_stability_v2.py` | Estabilidad con 20 misiones alternando acciones |
| `scripts/build_native_capabilities.py` | Construye el manifiesto de capacidades validadas (`outputs/native_capabilities_v2.json`) |
| `scripts/validate_teacher.py` | Valida el instructor en el motor nativo, sin red neuronal (42/42) |
| `scripts/validate_checkpoint_io_live.py` | Auditoría real de guardar, detener y reanudar el entrenador |
| `scripts/evaluate_checkpoint.py` | Evaluación independiente sobre validación reservada |

## Validaciones del cerebro (sin juego)
| Script | Propósito |
|---|---|
| `scripts/validate_brain.py` | Dinámica del grafo completo, reproducibilidad, intervención de entrada y benchmark |
| `scripts/validate_learning_full.py` | STDP en el grafo completo y continuación exacta desde checkpoint |
| `scripts/validate_motor_cone_equivalence.py` | Equivalencia bit a bit de la elegibilidad restringida al cono motor, con tiempos |
| `scripts/validate_sensory_transfer_v3.py --schema --output` | Transferencia pareada de cada canal sensorial a las DN |
| `scripts/causal_probe.py` | Intervenciones offline pareadas desde un checkpoint neuronal |
| `scripts/calibrate_plasticity.py` | [histórico] Barrido de tasas de la R-STDP global v2 |
| `scripts/calibrate_motor_plasticity.py` | Calibración pareada de la plasticidad motora (v1 frente a v2 y tasas) |
| `scripts/probe_decodability.py --schema` | Sonda offline de decodificabilidad sensores/DN → acción |
| `scripts/evaluate_imitation_offline.py --schema` | Compara procedimientos de imitación con CV por episodio |

## Entrenamiento y operación
| Script | Propósito |
|---|---|
| `scripts/train_curriculum.py --hours --reset-deadline` | Entrenador del currículo v3 (vigente) |
| `scripts/start_learning.ps1 -Hours -ResetDeadline` | Arranca panel, juego, entrenador y supervisor |
| `scripts/supervise_learning.ps1` | Relanza el entrenador si muere antes del plazo |
| `INICIAR APRENDIZAJE.cmd` | Ventana nueva de 12 h (`-Hours 12 -ResetDeadline`) |
| `scripts/serve_lab.py` | Servidor del panel (uvicorn, `127.0.0.1:8766`) |
| `scripts/train_continuous.py` | [histórico] Entrenador piloto v2 |
| `scripts/run_local.py`, `scripts/run_experiment.py`, `scripts/start_background.ps1`, `Iniciar Fly Operator.cmd` | [histórico] Lazo cerrado del prototipo v1 sin aprendizaje |
| `scripts/apply_performance_profile.py` | Transición auditada del perfil de recursos |
| `scripts/measure_performance.py` | Muestreo de CPU, RAM y GPU de bajo costo |
| `scripts/audit_night.py`, `scripts/readiness_report.py` | Resúmenes de integridad y preparación |
| `scripts/archive_frames.py [--restore]` | Mueve capturas fuera de la cuota con manifiesto SHA-256 |
| `scripts/archive_interrupted_evaluation_v3.py` | Archiva filas de validación previas a la publicación transaccional |
| `scripts/migrate_checkpoint_semantics_v2.py` | Migración única y auditada al contrato de capacidades v2 |
| `scripts/benchmark_checkpoint_io.py` | Compara compresión sin pérdida de checkpoints reales |
| `scripts/run_v3_tests.py` | Corre pytest y publica `outputs/v3_test_validation.json` |

## Informes, video y panel
| Script | Propósito |
|---|---|
| `scripts/build_v32_report.py` | Informe v3.2 (`outputs/INFORME_APRENDIZAJE_V32.md`) |
| `scripts/build_v3_report.py`, `scripts/build_learning_report.py`, `scripts/finalize_report.py` | Informes v3.1, v2 y v1 |
| `scripts/export_video.py`, `scripts/export_evaluation_video.py`, `scripts/record_research_clip.py`, `scripts/complete_recording.py` | Video a 30 fps desde capturas con marcas de tiempo del motor |
| `scripts/package_release.py` | Empaqueta código, puente y evidencia, sin el juego ni datos crudos |
| `scripts/polish_dashboard.py`, `scripts/polish_resume_ui.py` | [histórico] Parches de UI ya aplicados |
| `dashboard/qa.mjs`, `dashboard/qa_continuity.mjs` | QA con Playwright del panel |

## Módulos de `src/`
| Módulo | Contenido |
|---|---|
| `bridge_client.py` | Transporte con el puente y ciclo de misión (espera de carga de UI) |
| `learning_env.py` | Entorno Gymnasium, recompensa, cortes e instructor |
| `learning_adapter.py` | Codificación sensorial, lectura DN, catálogo y máscara de acciones |
| `learning_brain.py` | LIF completo, STDP, plasticidad motora y checkpoints del cerebro |
| `learning_policy.py` | Actor lineal, imitación, actualización interna y contribuciones |
| `learning_curriculum.py` | PPO por episodio, demostraciones y evaluación transaccional |
| `learning_checkpoint.py` | Checkpoints de dos generaciones y hash semántico |
| `learning_runtime.py` | Guardia de recursos, candado del entrenador y estado en vivo |
| `offline_replay.py` | Repetición offline y CV ridge agrupada |
| `lab_store.py`, `lab_history.py`, `lab_api.py` | Registro SQLite, historial y API del panel |
| `lossless_archive.py`, `runtime_config.py`, `windows_session.py` | Utilidades |
| `flybrain.py` | Simulador del prototipo v1 |

Pruebas: `tests/` (91 casos), `pytest -q` con `.venv-learning`.
