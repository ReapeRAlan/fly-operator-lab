"""Build the protocol 3.2 report from recorded evidence files in outputs/ and the live campaign database."""
from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3,math
ROOT=Path(__file__).resolve().parents[1]

def read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}

pct=lambda v:'—' if v is None else f'{100*v:.1f} %'

def wilson(success,n):
    if not n:return '—'
    z=1.959963984540054;p=success/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f'{max(0,c-r):.1%}–{min(1,c+r):.1%}'

def main():
    cfg=read(ROOT/'config/learning.json');run=cfg.get('experiment_id');out=ROOT/'outputs'
    cone=read(out/'motor_cone_equivalence.json');p31=read(out/'decodability_probe_v31.json');p32=read(out/'decodability_probe_v32.json')
    t31=read(out/'sensory_transfer_v3.json');t32=read(out/'sensory_transfer_v32.json');i31=read(out/'imitation_offline_v31.json');i32=read(out/'imitation_offline_v32.json')
    cal=read(out/'motor_plasticity_calibration_v32.json');tests=read(out/'v3_test_validation.json');schema=read(ROOT/cfg.get('sensory_schema',''))
    status=read(ROOT/'work/learning/status.json')
    changed=lambda t:[x for row in t.get('goal_summary',[]) for x in row.get('changed_each_seed',[])]
    rows=[]
    try:
        db=sqlite3.connect(f'file:{(ROOT/"work/learning/experiments.sqlite").as_posix()}?mode=ro',uri=True)
        rows=db.execute('SELECT condition,seed,stage,split,count(*),sum(success) FROM episodes WHERE run=? GROUP BY 1,2,3,4 ORDER BY 1,2,3,4',(run,)).fetchall();db.close()
    except sqlite3.Error:pass
    calibration=['| Regla | Tasa | Δ log π acreditada | Decisiones mejoradas | Cambio medio de ganancia por episodio | En límites |','|---|---:|---:|---:|---:|---:|']
    for r in cal.get('results',[]):
        calibration.append(f"| `{r['rule']}` | {r['learning_rate']:g} | {r['mean_delta_log_pi_credited']:+.4f} | {pct(r['fraction_improved'])} | {r['mean_abs_gain_change_per_episode']:.2e} | {pct(r['fraction_at_bounds'])} |")
    results=['| Condición | Semilla | Habilidad | Partición | Episodios | Éxitos | IC 95 % Wilson |','|---|---:|---|---|---:|---:|---|']
    results+=[f'| {c} | {s} | {st} | {sp} | {n} | {k or 0} | {wilson(k or 0,n)} |' for c,s,st,sp,n,k in rows] or ['| — | — | — | — | 0 | 0 | — |']
    probe=lambda p,k:p.get(k,{})
    lines=[f'# Fly Operator v{cfg.get("semantic_protocol_version")}: rendimiento, interfaz sensorial y plasticidad corregida','',
      f'Generado: {datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}. Campaña: `{run}`. La campaña v3.1 se conserva intacta como control.','',
      '## 1. Rendimiento: elegibilidad restringida al cono motor','',
      f'La regla causal motora solo acredita aristas que entran en las 1,314 descendentes ({cone.get("motor_cone_edges","—"):,} aristas). El integrador ya no etiqueta el resto del grafo; la dinámica de las {cone.get("neurons","—"):,} neuronas y {cone.get("edges","—"):,} aristas no cambia.','',
      f'- Equivalencia bit a bit sobre el grafo real (`{cone.get("checkpoint","—")}`, {cone.get("ticks","—")} ticks): **{"aprobada" if cone.get("passed") else "no aprobada"}**. Incluye voltajes, corrientes, refractarios, cola de retardo, trazas, ganancias, spikes por tick, instantáneas de elegibilidad motora y actualizaciones.',
      f'- Tick plástico: {cone.get("full_warm_median_seconds",float("nan")):.2f} s → **{cone.get("restricted_warm_median_seconds",float("nan")):.2f} s** (mediana; **{cone.get("speedup_warm_median",float("nan")):.1f}×**). Aristas rastreadas al final: {cone.get("full_active_edges_end","—"):,} → {cone.get("restricted_active_edges_end","—"):,}.','',
      '## 2. Interfaz sensorial v3.2','',
      '- Mismos 98 canales, mismas 1,568 compuertas excitatorias y mismo ranking anatómico. Cambia el **orden de asignación**: dirección y proximidad del objetivo reciben las rutas visuales más fuertes y el estado dinámico las rutas corporales más fuertes. En v3.1, el objetivo recibía rutas débiles y health/ammo, casi constantes, las más fuertes.',
      f'- Sectores con afinación von Mises (κ={schema.get("encoding",{}).get("sector_kappa","—")}); objetivo siempre saliente; `goal_distance` codifica proximidad exp(−d/{schema.get("encoding",{}).get("goal_proximity_m","—")} m); ganancia baja para estados de variación lenta.','',
      '**Sonda de decodificación** (repetición en lazo abierto de episodios `move` grabados; CV ridge agrupada por episodio con penalización elegida por CV interna; trayectorias duplicadas excluidas):','',
      '| Esquema | Episodios | Decisiones | Sensores → acción (±45°) | DN → acción exacto | DN → acción ±45° | Spikes / 50 ms |','|---|---:|---:|---:|---:|---:|---:|',
      *[f'| {name} | {p.get("unique_episodes","—")} | {p.get("decisions","—")} | {pct(probe(p,"sensors_to_action").get("within_45deg"))} | {pct(probe(p,"descending_to_action").get("exact"))} | {pct(probe(p,"descending_to_action").get("within_45deg"))} | {p.get("activity",{}).get("spikes_per_50ms_mean",0):,.0f} |' for name,p in (('v3.1',p31),('v3.2',p32))],'',
      f'- Criterio preregistrado DN → acción ±45° ≥ 90 %: **{"cumplido" if probe(p32,"descending_to_action").get("within_45deg",0)>=.9 else "no cumplido"}**.',
      f'- Fracción de DN ≥360 Hz: v3.1 {pct(p31.get("activity",{}).get("dn_near_ceiling_fraction"))}, v3.2 {pct(p32.get("activity",{}).get("dn_near_ceiling_fraction"))}. No bajó; ya era marginal y no limitaba la decodificación.',
      f'- Transferencia pareada: v3.1 {t31.get("distinct_goal_signatures","—")}/8 firmas, {min(changed(t31),default="—")}–{max(changed(t31),default="—")} DN cambiadas por sector; v3.2 **{"aprobada" if t32.get("passed") else "no aprobada"}**, {t32.get("distinct_goal_signatures","—")}/8 firmas, **{min(changed(t32),default="—")}–{max(changed(t32),default="—")}** DN cambiadas.','',
      'Límite: la sonda mide señal disponible para un lector lineal sobre trayectorias grabadas; no mide competencia en lazo cerrado.','',
      '## 3. Actor por imitación','',
      '| Features | Procedimiento v3.1 exacto / ±45° | Procedimiento v3.2 exacto / ±45° |','|---|---:|---:|',
      *[f'| {name} | {pct(i.get("v31_imitation",{}).get("exact"))} / {pct(i.get("v31_imitation",{}).get("within_45deg"))} | {pct(i.get("v32_imitation",{}).get("exact"))} / {pct(i.get("v32_imitation",{}).get("within_45deg"))} |' for name,i in (('v3.1',i31),('v3.2',i32))],'',
      'La hipótesis de subajuste del actor fue **refutada**: el procedimiento v3.1 alcanza el techo lineal. El optimizador separado con parada temprana generalizó peor y **no se adopta** (el código queda disponible, desactivado). La mejora del actor proviene de la interfaz sensorial.','',
      '## 4. Plasticidad causal motora v2','',
      '`causal_motor_rstdp_v1` movía cada ganancia en la dirección del crédito sin considerar el signo presináptico; en aristas inhibitorias (37.4 % del cono motor) eso reduce la actividad de la descendente acreditada. `causal_motor_rstdp_v2` multiplica el cambio por el signo presináptico; el signo y la anatomía no cambian.','',
      'Calibración pareada (mismos episodios y semillas; crédito positivo a la acción del instructor con magnitudes de error registradas en v3.1):','',*calibration,'',
      f'Tasa seleccionada por el criterio preregistrado: **{cal.get("selected_learning_rate")}**. Tasa configurada: **{cfg.get("motor_plasticity_learning_rate")}**. Criterio: {cal.get("criterion","—")}.','',
      *(['**Resultado negativo.** Ninguna tasa cumplió el criterio. Con tasas bajas, los cambios de ganancia son minúsculos y el cambio de log π acreditado es indistinguible del azar (~50 % de decisiones mejoradas). Con 1e-2, la política empeora en ambas reglas: mover eficacias desplaza las features descendentes fuera de la distribución con la que se ajustó el actor, y ese efecto domina al crédito. En lazo abierto la prueba no distingue v2 de v1; v2 se mantiene porque su dirección es correcta por construcción (prueba unitaria sobre una sinapsis aislada). Se conserva la tasa v3.1 (la menos disruptiva). La condición `internal` sigue en el protocolo como control; no se espera aprendizaje atribuible a ella con esta regla.',''] if cal and cal.get('selected_learning_rate') is None else []),
      '## 5. Operación','',
      f'- Plazo del piloto persistente en `schedule.json`; un reinicio no lo extiende (`-ResetDeadline` lo reinicia explícitamente). Plazo actual: {datetime.fromtimestamp(status["pilot_deadline"]).isoformat(timespec="minutes") if status.get("pilot_deadline") else "—"}.',
      f'- RAM con histéresis: pausa < {cfg.get("minimum_available_ram_gb")} GB, reanuda ≥ {cfg.get("resume_available_ram_gb")} GB.',
      '- Supervisor (`scripts/supervise_learning.ps1`, iniciado por `start_learning.ps1`): si el entrenador termina sin un paro intencional antes del plazo, lo relanza (máximo 6 veces) desde el último checkpoint, conserva los registros del fallo como `work/learning/crash_*` y anota en `work/learning/supervisor.log`. `faulthandler` registra fallos nativos.',
      '- Capturas v1/v2 (3.3 GiB) archivadas fuera de la cuota en `archive/frames` con manifiesto SHA-256 (`scripts/archive_frames.py --restore` las devuelve).',
      '- SQLite ya no repite las 100 etiquetas del catálogo por paso; el historial las reconstruye. El hash del actor se calcula una vez por episodio.',
      f'- Pruebas automatizadas: **{tests.get("tests","—")}**, aprobadas: {tests.get("passed","—")}.','',
      '## 6. Resultados de la campaña v3.2','',*results,'',
      '## 7. Límites','',
      '- GPU: no se adoptó. El integrador por eventos procesa ~200 spikes por paso de 0.1 ms; una versión GPU requeriría un kernel CUDA completo, una nueva validación y perdería reproducibilidad bit a bit con los checkpoints CPU.',
      '- Las sondas son de lazo abierto; la competencia autónoma solo se afirma con la validación reservada del currículo.',
      '- La dinámica LIF, los signos de neurotransmisor y la interfaz son supuestos de ingeniería documentados, no fisiología validada.']
    path=out/'INFORME_APRENDIZAJE_V32.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(path)

if __name__=='__main__':main()
