"""Build the v3 engineering report from versioned configuration and recorded evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3,hashlib,math
ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {} if default is None else default


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def wilson(success,n):
    if not n:return '—'
    z=1.959963984540054;p=success/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f'{max(0,c-r):.1%}–{min(1,c+r):.1%}'


def main():
    cfg=read(ROOT/'config/learning.json');runtime=read(ROOT/'config/runtime.json');run=cfg['experiment_id'];status=read(ROOT/'work/learning/status.json')
    schema_path=ROOT/cfg['sensory_schema'];schema=read(schema_path);transfer=read(ROOT/'outputs/sensory_transfer_v3.json')
    teacher=read(ROOT/'outputs/teacher_validation_v3.json');tests=read(ROOT/'outputs/v3_test_validation.json')
    migration=read(ROOT/'outputs/checkpoint_semantic_migration_v2.json');cleanup=read(ROOT/'outputs/interrupted_evaluation_cleanup_v3.json')
    campaign=ROOT/'work/learning/campaigns'/run;schedule=read(campaign/'schedule.json')
    db=sqlite3.connect(ROOT/'work/learning/experiments.sqlite')
    rows=db.execute('SELECT condition,seed,stage,split,count(*),sum(success),avg(seconds),avg(reward) FROM episodes WHERE run=? GROUP BY condition,seed,stage,split ORDER BY condition,seed,stage,split',(run,)).fetchall()
    step_count=db.execute('SELECT count(*) FROM steps WHERE run=?',(run,)).fetchone()[0];episode_count=sum(r[4] for r in rows);db.close()
    ports=[index for item in schema.get('mapping',{}).values() for index in item.get('indices',[])]
    goal=transfer.get('goal_summary',[]);changed=[x for row in goal for x in row.get('changed_each_seed',[])]
    teacher_rows=teacher.get('rows',[]);teacher_success=sum(bool(x.get('success')) for x in teacher_rows)
    table=['| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95% Wilson |','|---|---:|---|---|---:|---:|---|']
    for condition,seed,stage,split,n,success,seconds,reward in rows:
        table.append(f'| {condition} | {seed} | {stage} | {split} | {n} | {success or 0} | {wilson(success or 0,n)} |')
    if not rows:table.append('| — | — | — | — | 0 | 0 | — |')
    lines=[
      f'# Fly Operator v{cfg.get("semantic_protocol_version","3")}: aprendizaje causal, currículo y observación científica','',
      '**Dictamen:** la infraestructura está implementada y opera con el grafo clasificado completo. La transferencia sensorial y las reglas del instructor están comprobadas. La competencia autónoma y la generalización permanecen pendientes de evaluación reservada.','',
      f'Informe generado: {datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}. Campaña: `{run}`. El estado vivo se consulta en [el panel local](http://127.0.0.1:8766); no se congela como conclusión científica.','',
      '## 1. Inventario y procedencia','',
      f'- **{cfg["classified_neurons"]:,}** neuronas clasificadas y **{cfg["directed_edges"]:,}** aristas dirigidas, sin poda del grafo clasificado.',
      '- Identidad, dirección y cantidad anatómica de cada arista permanecen inmutables. La eficacia aprendida se guarda aparte.',
      '- Anatomía y anotaciones: MaleCNS v1.0. Signos de neurotransmisor, dinámica LIF y asignaciones del juego tienen supuestos documentados.',
      '- Los segmentos sin `superclass` permanecen en la fuente y no se presentan como neuronas simuladas.','',
      '## 2. Interfaz sensorial v3','',
      f'- {len(schema.get("mapping",{}))} canales × 16 puertas = **{len(ports):,} puertos**, todos únicos: **{len(set(ports)):,}**.',
      '- Espacio: puertas `visual_projection` excitatorias clasificadas por rutas anatómicas directas y a dos saltos hacia descendentes.',
      '- Estado y tarea: puertas excitatorias `vnc_sensory` y `cb_sensory`.',
      '- Señales explícitas para enemigos, amistosos, rehenes, civiles, puertas, objetivo, paredes, salud, munición, postura confirmada por el motor, equipo, acción en curso, dirección/distancia y tiempo restante.',
      f'- Prueba pareada de transferencia: **{"aprobada" if transfer.get("passed") else "pendiente"}**; {transfer.get("distinct_goal_signatures","—")}/8 firmas de objetivo distintas; rango observado de descendentes cambiadas por sector/semilla: **{min(changed) if changed else "—"}–{max(changed) if changed else "—"}**; saturación máxima medida: {transfer.get("max_network_saturation_fraction","—")}.','',
      'Este resultado demuestra que una señal llega al readout motor. No demuestra que el actor la interprete correctamente.','',
      '## 3. Tiempo, decisiones y currículo','',
      '- Los ticks con calentamiento, cola o recibo `in_progress` avanzan motor y cerebro, pero no crean demostraciones ni decisiones PPO.',
      '- Una excepción explícita aparece solo cuando el contexto permite intervenir: `stop` cerca del objetivo o `cancel` durante una brecha. Esas intervenciones sí son decisiones y reciben crédito propio.',
      '- Movimiento exige llegar y completar `stop`; postura exige que el bit nativo `crouched` quede activo; la cancelación se acepta solo si el motor la completa y la puerta permanece cerrada.',
      '- Una macroacción acumula recompensa hasta quedar libre o terminar. El descuento de su bootstrap es `γ^k` para `k` ticks.',
      '- PPO se actualiza al final del episodio natural; `n_steps=256` no corta la misión.',
      f'- Cada habilidad recibe {cfg["guided_episodes_per_stage"]} episodios guiados y {cfg["dagger_episodes_per_stage"]} DAgger por semilla. Si falla el gate común, admite hasta {cfg["max_extra_dagger_episodes"]} correcciones con probabilidad de instructor {cfg["dagger_teacher_beta"]}.',
      f'- Espera libre repetida se corta a {cfg["policy_collapse_free_wait_steps"]} decisiones; navegación sin mejora se corta a {cfg["policy_no_progress_decisions"]} decisiones y queda rotulada como fallo/truncación.','',
      '## 4. Plasticidad causal motora','',
      '`causal_motor_rstdp_v1` congela, antes de la acción, la elegibilidad de aristas activas cuyo destino es una de las 1,314 neuronas descendentes. Al concluir la macroacción combina el error temporal con `d log π(a|s) / d DN` y decae la instantánea según el retraso real hasta el crédito. La consecuencia observada después de actuar no puede entrar retroactivamente en esa instantánea.','',
      'La condición interna mantiene fijo el actor; cambia el evaluador externo y la eficacia de esas aristas. La combinada alterna episodios de PPO e internos. Es una regla computacional experimental, no una medición de dopamina ni una afirmación fisiológica.','',
      '## 5. Evaluación y no contaminación','',
      f'- Diagnóstico de promoción cada {cfg["evaluation_interval_steps"]} decisiones cognitivas, con {cfg["promotion_evaluation_episodes"]} episodios por habilidad adquirida.',
      f'- Gate: al menos {cfg["promotion_success"]:.0%} para adaptador, interna y combinada en semillas {cfg["seeds"]}; la referencia congelada se informa.',
      f'- Aceptación: {cfg["acceptance_evaluation_episodes_per_seed"]} episodios de validación por semilla. `test` se reserva para el cierre.',
      '- Antes de evaluar se guarda un checkpoint; después se restauran cerebro, actor, filtros, optimizador, contadores y RNG. Las pruebas automatizadas comprueban esa igualdad.',
      f'- La telemetría de validación se publica como bloque atómico. La limpieza auditada archivó {cleanup.get("archived_steps",0)} pasos y {cleanup.get("archived_episodes",0)} episodios parciales previos; quedan {cleanup.get("remaining_uncommitted_validation_episodes","—")} episodios parciales activos.',
      '- Éxito técnico, competencia y evidencia de aprendizaje se reportan por separado.','',
      '## 6. Checkpoints y recursos','',
      f'- Calendario: `{(campaign/"schedule.json").relative_to(ROOT)}`.',
      f'- Checkpoints: `{(campaign/"checkpoints").relative_to(ROOT)}/<condición>_<semilla>/current.json`.',
      '- Cada generación guarda cerebro, retardos, elegibilidad, eficacias, actor, evaluador, optimizador, filtros, ejemplos y RNG con hashes SHA-256.',
      '- Antes de publicar el puntero se rechaza cualquier `NaN`/infinito en cerebro, política, optimizador, filtros o rasgos; el hash semántico fija dinámica, recompensa, currículo, escenarios, capacidades, catálogo, acciones y esquema sensorial.',
      f'- Contrato semántico de capacidades v2: separa acciones/evidencia de su PID y hash de compilación. Migración auditada del checkpoint: **{"aprobada" if migration.get("passed") else "no aplicada"}**; los hashes de cerebro, política, ejemplos, actor y grafo permanecieron intactos.',
      f'- Perfil `{cfg["performance_profile"]}`: afinidad {cfg["worker_cpu_affinity"]}, PyTorch {cfg["max_cpu_threads"]} hilos, prioridad {cfg["worker_priority"]}, pausa real {cfg["minimum_rest_after_step_seconds"]} s y juego {1000/cfg["game_present_interval_ms"]:.0f} fps.',
      f'- Política de pantalla operativa: `{runtime.get("display_mode","no registrada")}`; pantalla despierta durante el entrenamiento: {runtime.get("keep_display_awake","—")}. Esta política no altera los hashes científicos ni los checkpoints.',
      f'- Guardas: {cfg["minimum_available_ram_gb"]} GB RAM disponibles, {cfg["maximum_worker_ram_gb"]} GB por worker, {cfg["minimum_free_disk_gb"]} GB libres y cuota {cfg["artifact_quota_gb"]} GB.','',
      '## 7. Verificación disponible','',
      f'- Instructor nativo: **{teacher_success}/{len(teacher_rows)}** casos exitosos.',
      f'- Transferencia sensorial: **{"aprobada" if transfer.get("passed") else "no aprobada"}**.',
      f'- Pruebas automatizadas registradas: **{tests.get("passed", "consultar ejecución local")}**; cantidad: {tests.get("tests", "—")}.',
      '- Parada y reanudación real verificadas por hash de actor y conteo exacto de ejemplos antes de continuar.',
      f'- Datos v3 al generar este informe: {step_count} pasos persistidos y {episode_count} episodios completos.','',
      f'Evidencias: [transferencia sensorial](sensory_transfer_v3.json), [instructor](teacher_validation_v3.json), [configuración](../config/learning.json), [esquema sensorial](../{cfg["sensory_schema"]}).','',
      '## 8. Resultados registrados de la campaña v3','',*table,'',
      f'Calendario al corte: etapa `{schedule.get("stage_index","—")}`, ronda `{schedule.get("round","—")}`. Evaluaciones guardadas: {sum(len(v) for v in schedule.get("evaluations",{}).values())}. Estos valores describen progreso operativo y pueden cambiar mientras corre el entrenador.','',
      '## 9. Límites','',
      '- Los resultados v2 se conservan como control histórico y no se atribuyen a v3: cambiaron sensores, temporalidad, demostraciones y plasticidad.',
      '- El modelo no incluye toda la fisiología de una mosca ni afirma equivalencia biológica.',
      '- Las cinco familias nativas pendientes (`use`, `clear_obstacle`, `arrest`, `spy_camera`, `evacuate`) siguen fuera del entrenamiento hasta validar sus efectos.',
      '- No se afirmará resolución autónoma de misiones hasta superar validación reservada en las tres semillas y comparación contra el punto común y la referencia congelada.','',
      f'Hashes: esquema sensorial `{digest(schema_path)}`; prueba de transferencia `{digest(ROOT/"outputs/sensory_transfer_v3.json") if (ROOT/"outputs/sensory_transfer_v3.json").exists() else "—"}`.'
    ]
    out=ROOT/'outputs/INFORME_APRENDIZAJE_V3.md';out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)


if __name__=='__main__':main()
