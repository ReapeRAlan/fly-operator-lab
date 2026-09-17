# Terceros y atribuciones

| Componente | Uso | Licencia | Incluido en el repo |
|---|---|---|---|
| **MaleCNS v1.0**, Janelia FlyEM (https://male-cns.janelia.org/download/) | Anatomía: neuronas, conexiones y neurotransmisor predicho | CC-BY 4.0, según la página oficial | No: se descarga con `scripts/acquire.py dataset` |
| **Drosophila_brain_model**, Shiu et al. 2024 (https://github.com/philshiu/Drosophila_brain_model) | Referencia de la dinámica LIF | MIT | No: se descarga con `acquire.py toolchain` |
| **Cuaderno 2025malecns**, flyconnectome (https://github.com/flyconnectome/2025malecns) | Criterio de conexiones por superclase | Ver su repositorio | No |
| **MinHook** v1.3.4 (https://github.com/TsudaKageyu/minhook) | Ganchos del puente nativo | BSD-2-Clause | No: `acquire.py toolchain` |
| **nlohmann/json** v3.12.0 (https://github.com/nlohmann/json) | JSON en el puente | MIT | No: `acquire.py toolchain` |
| **stb_image_write** (https://github.com/nothings/stb) | Capturas PNG | MIT o dominio público | No: `acquire.py toolchain` |
| **Ghidra** 12.1.3 (NSA) y **OpenJDK 25** (Adoptium) | Análisis del binario local | Apache-2.0 / GPLv2 + Classpath Exception | No: `acquire.py toolchain` |
| Python: numpy, numba, scipy, pyarrow, torch, gymnasium, stable-baselines3, sb3-contrib, fastapi, uvicorn… | Simulación, aprendizaje y API | Ver `requirements-learning.lock.txt` | No (pip) |
| Panel: React, Vite, lucide-react, Playwright | Interfaz | MIT / Apache-2.0 / ISC | No (npm); `dashboard/dist` es el build propio |

## Door Kickers 2
**Door Kickers 2** es propiedad de KillHouse Games. Este repositorio **no incluye** el juego, sus archivos, sus plantillas XML ni código descompilado. Para usar el laboratorio se necesita una copia legítima (Steam, v1.12). El puente trabaja solo con un perfil aislado en `work/profile` y no modifica el ejecutable ni el perfil de Steam.

## Licencia del código propio
No se ha elegido todavía una licencia para el código de este repositorio. Mientras no exista un archivo `LICENSE`, se aplican los derechos de autor por defecto.
