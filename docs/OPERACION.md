# Operación

## Requisitos
- Windows x64 con **Door Kickers 2 v1.12** de Steam. El puente rechaza cualquier otro ejecutable: SHA-256 `7f873f34…d747`.
- Python 3.11 (`.venv-learning`, dependencias en `requirements-learning.lock.txt`) y Python 3.9 (`.venv`, prototipo).
- Visual Studio con MSVC para compilar el puente, y Node.js para compilar el panel.
- ~2 GB para el conectoma procesado; 16 GB de RAM recomendados.

## Reproducir desde cero
```powershell
python -m venv .venv-learning
.venv-learning\Scripts\python -m pip install -r requirements-learning.lock.txt
.venv-learning\Scripts\python scripts/acquire.py dataset      # conectoma MaleCNS v1.0 (acceso público)
.venv-learning\Scripts\python scripts/acquire.py toolchain    # MinHook, json.hpp, stb, Ghidra, JDK, modelo Shiu
.venv-learning\Scripts\python scripts/prepare_connectome.py
.venv-learning\Scripts\python scripts/setup_profile.py
.venv-learning\Scripts\python scripts/prepare_learning.py
# build_profile.h requiere repetir el análisis del binario si cambia la versión del juego
powershell -File scripts/build_native.ps1
.venv-learning\Scripts\python scripts/build_sensory_schema_v32.py
cd dashboard; npm install; npm run build; cd ..
.venv-learning\Scripts\python -m pytest -q
```

## Entrenar una noche
- Doble clic en **`INICIAR APRENDIZAJE.cmd`**. Abre una ventana nueva de 12 h (`start_learning.ps1 -Hours 12 -ResetDeadline`) y arranca:
  1. panel (`127.0.0.1:8766`);
  2. juego con puente;
  3. entrenador;
  4. supervisor.
- **El plazo se guarda en `schedule.json`.** Reiniciar sin `-ResetDeadline` conserva el plazo; si ya venció, el entrenador termina enseguida.
- **Guardar y detener piloto** en el panel escribe un checkpoint y termina el entrenador y la supervisión.

### Recursos (`config/learning.json`, editables en caliente)
| Clave | Valor actual | Efecto |
|---|---|---|
| `minimum_available_ram_gb` | 1.0 | Pausa (guardando antes) por debajo de este valor |
| `resume_available_ram_gb` | 1.5 | Reanuda al superar este valor |
| `maximum_worker_ram_gb` | 4 | Límite del proceso entrenador |
| `minimum_free_disk_gb` / `artifact_quota_gb` | 8 / 8 | Disco libre y cuota de registros |

El entrenador relee estos límites cada 10 s. Afinidad (`worker_cpu_affinity`), hilos y prioridad se aplican al arrancar.

## Ver el panel desde otro dispositivo
El panel escucha solo en `127.0.0.1`. Para verlo en la red Tailscale sin abrir el firewall:
```powershell
tailscale serve --bg --yes --tcp 8766 tcp://127.0.0.1:8766
# desactivar: tailscale serve --tcp=8766 off
```
Después abre `http://<IP-de-Tailscale-de-la-PC>:8766/`. Desde fuera es **de solo lectura**: pausar y detener responden 403, a propósito.

## Problemas conocidos
| Síntoma | Explicación y solución |
|---|---|
| El juego muestra "Planning Mode" y el reloj casi no avanza | Es normal: el juego se pausa entre pasos de 50 ms mientras la red calcula, así que corre ~8 veces más lento que el tiempo real |
| La ventana del juego se cerró | Cerrar la ventana detiene la sesión; el supervisor reabre el juego y retoma desde el checkpoint en ~1 min |
| Crash al abrir el juego (`Game::RestartMap` en el log) | Se pidió una misión durante la carga inicial. Corregido: el cliente espera `Loading UI took` en el log de la sesión |
| El entrenador aparece "en pausa" | RAM disponible bajo el mínimo; cierra aplicaciones o ajusta los límites de la tabla |
| `Checkpoint semantic protocol mismatch` | Cambió algo científico (protocolo, esquema, acciones, dinámica); hace falta una campaña nueva |
| Registros de fallos | `work/learning/supervisor.log`, `work/learning/crash_*`, `work/bridge.log`, `work/profile/KillHouseGames/DoorKickers2/log.txt` y `CrashDump_*.zip` |

## Archivos que no se publican
Viven en la máquina local y el `.gitignore` los excluye:
- `work/`: perfil del juego, checkpoints, SQLite, capturas, descompilación y builds;
- `archive/`;
- `data/raw` y `data/processed/*.npy|*.bin|*.feather`;
- `data/learning/equipment_catalog.json`: derivado de los XML del juego;
- `third_party/`: se descarga con `acquire.py`;
- entornos virtuales y `node_modules`.
