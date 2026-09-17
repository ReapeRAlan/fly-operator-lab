"""Regenerate an evidence-bounded comparative report from actual local records."""
from pathlib import Path
import sys,json,math,datetime,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lab_store import connect,atomic_json,RUNTIME,file_hash

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-w),min(1,c+w)]

def main():
    db=connect();groups=[];run=read(ROOT/'config/learning.json').get('experiment_id','pilot-v2')
    for row in db.execute('SELECT condition,seed,stage,split,count(*),sum(success),sum(native_victory),avg(seconds),avg(reward) FROM episodes WHERE run=? GROUP BY condition,seed,stage,split',(run,)):
        c,seed,stage,split,n,k,w,seconds,reward=row
        groups.append(dict(condition=c,seed=seed,stage=stage,split=split,episodes=n,successes=k,native_victories=w,success_rate=k/n,wilson95=wilson(k,n),mean_seconds=seconds,mean_reward=reward))
    db.close();status=read(RUNTIME/'status.json');cap=read(ROOT/'outputs/native_capabilities_v2.json');full=read(ROOT/'outputs/learning_full_validation.json');train=read(ROOT/'outputs/training_validation_v2.json');probe=read(ROOT/'outputs/causal_probe_v2.json')
    atomic_json(ROOT/'outputs/comparison_v2.json',{'generated':datetime.datetime.now().isoformat(),'groups':groups,'acceptance_requires':{'seeds':[7,19,43],'heldout_episodes_per_skill_per_seed':30,'success_threshold':.8},'full_acceptance_met':False,'status_snapshot':status})
    lines=['# Fly Operator v2: aprendizaje y observación científica','',
      '**Resultado:** el mecanismo de aprendizaje y el panel funcionan en pruebas locales con la red clasificada completa. El piloto está preparado para ejecutar enseñanza y práctica reanudable. La mejora sostenida en misiones reservadas todavía no está demostrada.','',
      f'Informe generado: {datetime.datetime.now().isoformat(timespec="seconds")}. Estado observado: **{status.get("state")}**; fase **{status.get("phase","inicialización")}**, semilla **{status.get("seed","pendiente")}**. El estado cambia durante el piloto; consultar [panel local](http://127.0.0.1:8766).','',
      '## 1. Lo implementado y lo comprobado','',
      '| Componente | Evidencia y alcance |','|---|---|',
      '| Red completa clasificada | 166,700 nodos, 25,582,938 aristas dirigidas; sin poda adicional. Hash de grafo: `'+full['graph_hash']+'`. |',
      '| Plasticidad | R-STDP con eficacias separadas, signo conservado y límites 0.25–4. La prueba cambió '+str(train['internal_update']['changed_edges'])+' eficacias; el actor permaneció idéntico durante esa actualización. |',
      '| Adaptador | Actor lineal sobre 1,314 neuronas descendentes × filtros de 100/500/2,000 ms; imitación y MaskablePPO modificaron sus pesos en una prueba dentro del juego. |',
      '| Reanudación | Restauración exacta de actividad, cola de retardos, RNG neuronal, pesos, filtros y RNG de PyTorch. Optimizador incluido y restaurado. |',
      '| Puente | 20 ciclos consecutivos de acciones nativas, sin duplicación de secuencias ni cierre. Estos ciclos usaron órdenes de prueba, no aprendizaje neuronal. |',
      '| Control | 16 familias habilitadas: '+', '.join(cap['validated_actions'])+'. |',
      '| Equipo | 848 plantillas XML indexadas, atributos completos, relaciones de compatibilidad y parámetros efectivos del inventario. Tres conjuntos de equipo probados. |',
      '| Interfaz | Cinco vistas, búsqueda por bodyId/tipo/región anotada, eficacia por arista, trazas observadas, probabilidades, contribuciones al logit y exportaciones. |',
      '| Video | Muestra de 0.9 segundos de cambio de arma por actor sin instructor; un solo caso, fuera de las estadísticas de aceptación. MP4 1920×1080/30. |','',
      'La ampliación **no cubre todavía todas las familias del motor**: `use`, `clear_obstacle`, `arrest`, `spy_camera` y `evacuate` permanecen deshabilitadas hasta verificar sus efectos. El catálogo XML completo no implica que cada objeto sea una acción entrenable. La cancelación validada corresponde a apertura con herramienta; los ciclos de arma conservan su disponibilidad nativa.','',
      'Pruebas: [red y checkpoint](learning_full_validation.json), [aprendizaje dentro del juego](training_validation_v2.json), [acciones](native_actions_v2.json), [equipo y cancelación](native_kits_v2.json), [20 ciclos del puente](stability_v2.json), [calibración](plasticity_calibration_v2.json), [intervenciones](causal_probe_v2.json).','',
      '## 2. Qué se modela','',
      'Se simulan todas las filas con superclass identificada, incluidas categorías provisionales. Los segmentos sin clasificación quedan en el grafo crudo original. Hay 516 registros Traced sin superclass fuera de este inventario; no se presentan como neuronas modeladas. Los 211,577 registros de anotaciones no equivalen al número de neuronas simuladas. El archivo de auditoría conserva además conexiones hacia segmentos fuera del inventario.','',
      'Datos anatómicos y predicciones de neurotransmisores proceden de [MaleCNS v1.0](https://male-cns.janelia.org/download/). La conectividad no proporciona por sí sola constantes de membrana, eficacia eléctrica, aprendizaje, visión del juego o conceptos de armas. Esas asignaciones se documentan como supuestos de ingeniería. El modelo LIF toma como referencia la implementación de [Shiu](https://github.com/philshiu/Drosophila_brain_model); su modelo original usa otro conectoma.','',
      'La codificación usa 74 canales × 16 neuronas sensoriales asignadas de forma fija: entidades visibles, puertas, objetivo anunciado, rayos contra geometría conocida, estado propio y etapa. El actor recibe únicamente 3,942 valores derivados de actividad descendente. No recibe coordenadas ni salud ni munición de enemigos ocultos. Las restricciones de acciones sí reflejan disponibilidad mecánica. La búsqueda de región usa `somaNeuromere`, cuando está anotada; no equivale a una auditoría completa de innervación por neuropilo.','',
      '## 3. Dónde se almacena el aprendizaje','',
      '- `work/learning/checkpoints/<condición>_<semilla>/current.json`: manifiesto de la generación válida y hashes SHA-256.',
      '- `<generación>/brain.npz`: eficacias por conexión, voltajes, estado sináptico, periodo refractario, retardos pendientes, huellas, elegibilidad, filtros de silenciamiento y RNG neuronal.',
      '- `<generación>/policy.pt`: actor, evaluador, optimizador, filtros temporales, RNG de PyTorch/NumPy/Python y contadores.',
      '- `work/learning/examples_<semilla>.npz`: ejemplos guiados y etiquetas agregadas en estados visitados por el aprendiz.',
      '- `work/learning/experiments.sqlite`: observaciones permitidas, decisiones, recompensas, eficacia actualizada, episodios y checkpoints.',
      '- `work/learning/schedule.json`: semillas preparadas, condición, ronda, etapa y resultados de validación.',
      '',
      'Los estados neuronales se pueden repetir exactamente fuera del juego. El heap del motor no se serializa: al reabrir el entrenador se conserva el aprendizaje y se reinicia el episodio interrumpido. Los hashes verifican integridad antes de deserializar un checkpoint local. Los datos anatómicos originales permanecen inmutables.','',
      '## 4. Protocolo de entrenamiento','',
      'Cada semilla (7, 19, 43) recibe 24 episodios guiados y 8 episodios con etiquetas del instructor en estados del aprendiz, siguiendo el principio de [DAgger](https://proceedings.mlr.press/v15/ross11a.html). Tras esa preparación se crea el punto de partida común para las cuatro condiciones. La imitación se actualiza después de cada episodio; los ejemplos previos se conservan.','',
      '| Condición | Parámetros que cambian en práctica |','|---|---|','| Congelada | Ninguno de los que determinan acciones. |','| Adaptador | Actor y evaluador mediante PPO; eficacia interna fija. |','| Interna | Eficacia interna mediante R-STDP y evaluador de recompensa; actor fijo. |','| Combinada | Alterna bloques de PPO y R-STDP; las representaciones internas permanecen fijas durante cada bloque PPO. |','',
      'La señal de modulación es el error temporal de un evaluador externo: recompensa + γV(siguiente) − V(actual), sin bootstrap en terminación real y con bootstrap cuando hay un corte externo. No se identifica esa señal con una medición de dopamina. Referencias: [R-STDP](https://www.izhikevich.org/publications/dastdp.htm), [MaskablePPO y su ausencia de política recurrente](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).','',
      'Las rondas asignan 256 decisiones por condición/semilla. Cuando un bloque termina a mitad de episodio se registra `truncated`, motivo `condition_switch_budget`, conservando el bootstrap. Esto limita cada tramo a 12.8 segundos; para estudiar misiones largas debe ampliarse `ppo.n_steps` y recalibrarse el presupuesto. La comparación registra tiempo simulado y real por separado.','',
      'Las evaluaciones se programan cada 4,096 decisiones: 30 mapas reservados por semilla y por habilidad adquirida, sin instructor ni plasticidad. Las nueve combinaciones de condiciones que aprenden × semillas deben alcanzar 80% y conservar habilidades anteriores para promover la etapa común. La referencia congelada se evalúa y se reporta, pero su fracaso no bloquea indefinidamente la promoción. Uno de cada cuatro episodios de práctica revisita habilidades anteriores. Los mapas test se reservan para evaluación final mediante el script independiente.','',
      'La recompensa combina éxito, derrota, costo de tiempo y diferencias de potencial hacia el objetivo. No recompensa disparos ni recargas por sí mismos. La bonificación potencial usa γΦ(siguiente)−Φ(actual), con potencial terminal cero; las garantías teóricas dependen de los supuestos expuestos en [Ng, Harada y Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).','',
      '## 5. Parámetros y calibración','',
      'Valores iniciales: paso 0.1 ms; decisión 50 ms; membrana 20 ms; sinapsis 5 ms; retardo 1.8 ms; reposo −52 mV; umbral −45 mV; refractario nominal 2.2 ms (separación mínima efectiva 2.3 ms con el orden de actualización); huellas 20 ms; elegibilidad 2 s; η=10⁻⁴; A+=1 y A−=1.05; eficacia ×0.25–4. PPO: tasa 3×10⁻⁴, γ=.999 y clip=.2. Las unidades y procedencia están en `/api/parameters`, `config/learning.json` y `src/learning_brain.py`.','',
      'Se probaron η de 10⁻⁵, 10⁻⁴ y 10⁻³ desde el mismo estado y ruido, con resultados finitos y eficacias dentro de límites. En esa ventana de 50 ms hubo 90.3% de neuronas silenciosas y 0.065% cerca de la frecuencia máxima del modelo. Esto caracteriza ese estímulo y esa ventana; no prueba calibración fisiológica ni estabilidad a largo plazo.','',
      f'La última prueba completa ocupó {full["rss_mib"]:.1f} MiB en el proceso de simulación aislada. Las memorias auxiliares se guardan en archivos mapeados y los estados de elegibilidad vacíos tienen codificación compacta. Los hashes de impulsos antes/después de la optimización coincidieron. El entrenador añade PyTorch, ejemplos y registros; el panel muestra su disponibilidad real de RAM.','',
      '## 6. Intervenciones y video','',
      'La repetición desde el mismo estado produjo exactamente los mismos impulsos. Cambiar entradas modificó las probabilidades (distancia L1 '+f'{probe["probability_l1_effect"]["altered_input"]:.6f}'+'); silenciar neuronas descendentes produjo L1 '+f'{probe["probability_l1_effect"]["silenced_descending"]:.6f}'+'. Son intervenciones computacionales fuera del juego; no demuestran causalidad biológica ni éxito táctico. Las contribuciones lineales mostradas en el panel son asociaciones exactas con el logit, no explicaciones causales completas.','',
      'El botón Capturar ventana guarda checkpoint previo y los impulsos del siguiente paso. Las conexiones consultadas registran huellas pre/post y elegibilidad. La serie de voltaje comienza al consultar una neurona y se separa al reiniciar su estado. Se exportan SVG, PNG, CSV, parámetros y un paquete ZIP de evidencia.','',
      '[Clip técnico MP4](FlyOperator_investigacion_v2_1080p30.mp4): 0.9 segundos, cambio de arma completado en un mapa reservado con un checkpoint de integración que combina imitación, PPO y una actualización R-STDP. El rótulo lo identifica como prueba combinada. El archivo [de trazabilidad](research_clip_trace_v2.json) conserva la condición interna original `adapter` y añade la corrección de clasificación; esa etiqueta antigua no debe usarse para atribuir el resultado a adaptación exclusiva. Una muestra no satisface 30 episodios × tres semillas.','',
      'Los fotogramas nativos se capturan cada 50 ms (20 fps), se mantienen/retemporizan a 30 fps y se escalan con bandas a 1080p. Las esperas de cómputo no entran en el video. El manifiesto del clip documenta resolución de origen, tiempo y conversión.','',
      '## 7. Uso y continuidad','',
      '```powershell','powershell -NoProfile -ExecutionPolicy Bypass -File D:\\FlyOperatorLab\\scripts\\start_learning.ps1 -Hours 24','```','',
      'Abrir http://127.0.0.1:8766. Pausar/Reanudar opera al terminar el paso actual; Guardar y detener termina con checkpoint. El juego permanece en el monitor secundario y usa el perfil aislado. Una sola instancia controla la canalización. Prioridad baja, perfil quiet_v1 con un procesador lógico, prioridad IDLE del entrenador, descanso mínimo de 750 ms por paso y presentación limitada a 20 fps; CPU para aprendizaje, mínimo 3 GB RAM disponible y 8 GB disco libre. La cuota de 8 GB considera registros, capturas y entregables. Las pausas por recursos se reanudan automáticamente al recuperarlos; errores de transporte o numéricos detienen con motivo y checkpoint cuando es posible.','',
      'Un fallo de carga observado al arrancar antes de completar recursos se corrigió retrasando la primera solicitud hasta al menos diez segundos desde el lanzamiento. Las pruebas posteriores pasaron; un equipo mucho más lento puede requerir revisar esa guarda. La semilla solicitada controla geometría y RNG neuronal. El motor publica su propia `map_seed`, que se registra; no se afirma reproducción exacta de toda la física del juego entre reinicios.','',
      'Herramientas: `scripts/evaluate_checkpoint.py` para evaluar un checkpoint con el entrenador detenido; `scripts/causal_probe.py` para intervenciones fuera del juego; `scripts/calibrate_plasticity.py`; `scripts/export_evaluation_video.py`; `scripts/build_learning_report.py` para actualizar este informe. Entorno independiente: `.venv-learning` / Python 3.11, dependencias en `requirements-learning.lock.txt`.','',
      '## 8. Resultados comparativos observados','',
      '| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95% Wilson |','|---|---:|---|---|---:|---:|---|']
    for g in groups:
        low,high=g['wilson95'];lines.append(f'| {g["condition"]} | {g["seed"]} | {g["stage"]} | {g["split"]} | {g["episodes"]} | {g["successes"]} | {low:.1%}–{high:.1%} |')
    if not groups:lines.append('| Sin episodios completos registrados todavía | — | — | — | 0 | — | — |')
    lines+=['','Los éxitos de enseñanza usan instructor y no cuentan como aprendizaje demostrado. Los intervalos describen los episodios observados; las semillas se reportan por separado y no se simula un tamaño de muestra mayor. Las pérdidas protegidas, consumibles, acciones rechazadas, duración y costo de cálculo están en los registros por episodio y paso.','',
      '**Aceptación completa pendiente:** las cuatro condiciones × tres semillas, 30 episodios reservados por habilidad, retención y misiones finales. El piloto y el panel permiten medirlo; todavía no hay evidencia para afirmar que aprendió a resolver las misiones del juego.']
    (ROOT/'outputs/INFORME_APRENDIZAJE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'version':2,'generated':datetime.datetime.now().isoformat(),'sources':{}}
    for folder in ('src','scripts','config','native','dashboard/src'):
        for path in (ROOT/folder).glob('*'):
            if path.is_file() and path.suffix in ('.py','.ps1','.cpp','.h','.json','.jsx','.css','.java'):manifest['sources'][str(path.relative_to(ROOT))]=file_hash(path)
    manifest['session']=read(ROOT/'work/session.json');manifest['graph_hash']=full['graph_hash'];manifest['status_snapshot']=status
    atomic_json(ROOT/'outputs/learning_manifest_v2.json',manifest)
    print(json.dumps({'report':str(ROOT/'outputs/INFORME_APRENDIZAJE.md'),'groups':len(groups),'status':status.get('state')}))
if __name__=='__main__':main()
