from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'dashboard/src/main.jsx';s=p.read_text(encoding='utf-8')
changes={
"stopped:'Detenido',error:":"stopped:'Detenido',paused_by_user:'En pausa',error:",
"<p>Contribuciones exactas al logit del actor; no demuestran causalidad por sí solas.</p>":"<p>Propuesta del actor: {l.decision?.action_labels?.[l.decision?.contributions_for_action??l.decision?.selected]||'—'}. Durante la enseñanza, la orden enviada puede ser distinta. Estas contribuciones no demuestran causalidad.</p>",
"<button className=\"btn\" onClick={()=>control(s.state==='paused'?'resume':'pause')}>":"<button className=\"btn\" disabled={!['running','paused','saving','preparing'].includes(s.state)} title={s.state==='stopped'?'Abre INICIAR APRENDIZAJE.cmd para iniciar el entrenador':undefined} onClick={()=>control(s.state==='paused'?'resume':'pause')}>",
}
for old,new in changes.items():
    if old not in s:raise ValueError('Expected UI text missing: '+old[:90])
    s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
