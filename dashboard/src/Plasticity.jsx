import React, { useEffect, useState } from "react";
import { AlertTriangle, GitBranch, RefreshCw } from "lucide-react";
import {withRun} from "./Provenance.jsx";

const api = async (path) => {
  const response = await fetch("/api/" + path);
  if (!response.ok) throw Error(await response.text());
  return response.json();
};
const number = (value, digits = 3) =>
  value == null
    ? "—"
    : Number(value).toLocaleString("es-MX", { maximumFractionDigits: digits });

function GainChart({ history }) {
  if (history.length < 2)
    return (
      <div className="empty">
        Esperando dos auditorías de gains para dibujar la tendencia.
      </div>
    );
  const width = 760,
    height = 250,
    left = 48,
    right = 18,
    top = 20,
    bottom = 32,
    all = history.flatMap((item) => [item.p05, item.p95, item.mean]),
    low = Math.min(...all),
    high = Math.max(...all),
    scale = (value) =>
      height -
      bottom -
      ((value - low) / Math.max(1e-9, high - low)) * (height - top - bottom),
    x = (index) =>
      left + (index * (width - left - right)) / Math.max(1, history.length - 1);
  const line = (key) =>
    history.map((item, index) => `${x(index)},${scale(item[key])}`).join(" ");
  return (
    <svg
      className="gain-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Tendencia de estabilidad de gains"
    >
      <polygon
        points={`${history.map((item, index) => `${x(index)},${scale(item.p95)}`).join(" ")} ${history
          .slice()
          .reverse()
          .map((item, index) => {
            const original = history.length - 1 - index;
            return `${x(original)},${scale(item.p05)}`;
          })
          .join(" ")}`}
        fill="#d8eee9"
        opacity=".7"
      />
      <polyline
        points={line("median")}
        fill="none"
        stroke="#208e89"
        strokeWidth="3"
      />
      <polyline
        points={line("mean")}
        fill="none"
        stroke="#738db4"
        strokeWidth="2"
        strokeDasharray="5 4"
      />
      <text x={left} y={height - 8}>
        revisión {history[0].revision}
      </text>
      <text x={width - right} y={height - 8} textAnchor="end">
        revisión {history.at(-1).revision}
      </text>
      <text x={left} y={14}>
        gain · percentiles 05-95
      </text>
    </svg>
  );
}
function GainHistogram({ latest }) {
  const bins = latest?.histogram || [],
    max = Math.max(1, ...bins.map((bin) => bin.count));
  if (!bins.length)
    return (
      <div className="empty">
        El histograma aparecerá en la próxima auditoría.
      </div>
    );
  return (
    <div className="gain-histogram">
      {bins.map((bin) => (
        <div className="gain-bin" key={bin.from}>
          <b style={{ height: `${Math.max(3, (100 * bin.count) / max)}%` }} />
          <span>{bin.from.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Plasticity({run}) {
  const [data, setData] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true,
      timer;
    const poll = async () => {
      try {
        const next = await api(withRun("plasticity",run));
        if (active) {
          setData(next);
          setError("");
        }
      } catch (e) {
        if (active) setError("No se pudo leer la auditoría de plasticidad");
      }
      if (active) timer = setTimeout(poll, 5000);
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [run]);
  if (error)
    return (
      <div className="notice" role="alert">
        {error}
      </div>
    );
  if (!data)
    return (
      <div className="panel">
        <div className="empty">Consultando plasticidad en vivo…</div>
      </div>
    );
  const latest = data.latest,
    alerts = data.alerts || [],
    auditMissing = data.archive_available === false;
  return (
    <div className="plasticity-view">
      <article className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">APRENDIZAJE EN VIVO</span>
            <h2>Estabilidad de plasticidad</h2>
            <p>
              {auditMissing ? 'Auditoría no archivada para esta campaña; no se reutilizan trazas de otra ejecución.' : 'Auditoría muestreada de gains · no sustituye una validación causal del modelo.'}
            </p>
          </div>
          <span className="tag">
            <RefreshCw size={13} />{" "}
            {latest ? `Revisión ${latest.revision}` : "Sin auditorías"}
          </span>
        </div>
        <div className="plasticity-kpis">
          <div>
            <span>Gain medio</span>
            <strong>{number(latest?.mean)}</strong>
            <small>mediana {number(latest?.median)}</small>
          </div>
          <div>
            <span>Dispersión</span>
            <strong>{number(latest?.std)}</strong>
            <small>
              p05 {number(latest?.p05)} · p95 {number(latest?.p95)}
            </small>
          </div>
          <div>
            <span>Suelo &lt; 0.01</span>
            <strong>
              {latest ? `${number(latest.below_floor * 100, 1)}%` : "—"}
            </strong>
            <small>{number(latest?.sample_size, 0)} aristas muestreadas</small>
          </div>
          <div>
            <span>Techo &gt; 10</span>
            <strong>
              {latest ? `${number(latest.above_ceiling * 100, 1)}%` : "—"}
            </strong>
            <small>ganancia potencialmente saturada</small>
          </div>
        </div>
        <div className="gain-chart-wrap">
          {auditMissing ? <div className="empty">Auditoría no archivada</div> : <GainChart history={data.history || []} />}
          <div className="chart-legend">
            <span>
              <i className="legend-band" />
              p05-p95
            </span>
            <span>
              <i className="legend-line" />
              mediana
            </span>
            <span>
              <i className="legend-dash" />
              media
            </span>
          </div>
        </div>
        <div className="gain-histogram-panel">
          <h3>Histograma de pesos muestreados</h3>
          {auditMissing ? <div className="empty">Sin histograma archivado.</div> : <GainHistogram latest={latest} />}
        </div>
      </article>
      <div className="plasticity-columns">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Alertas</h2>
              <p>Reglas de supervisión sobre la muestra actual.</p>
            </div>
            <AlertTriangle size={18} />
          </div>
          {alerts.length ? (
            <ul className="health-alerts">
              {alerts.map((alert) => (
                <li key={alert}>{alert}</li>
              ))}
            </ul>
          ) : (
            <div className="health-ok">
              Sin desviaciones de gains detectadas.
            </div>
          )}
          <dl className="health-details">
            <div>
              <dt>Condición</dt>
              <dd>{latest?.condition || "—"}</dd>
            </div>
            <div>
              <dt>Semilla</dt>
              <dd>{latest?.seed ?? "—"}</dd>
            </div>
          </dl>
        </article>
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Edges observados</h2>
              <p>Solo conexiones seleccionadas por el explorador neuronal.</p>
            </div>
            <GitBranch size={18} />
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Edge</th>
                  <th>Gain</th>
                  <th>Elegibilidad</th>
                  <th>Reloj</th>
                </tr>
              </thead>
              <tbody>
                {(data.traces || []).map((trace) => (
                  <tr key={trace.edge_index}>
                    <td>{trace.edge_index}</td>
                    <td>{number(trace.gain)}</td>
                    <td>{number(trace.eligibility)}</td>
                    <td>{number(trace.clock_ms, 1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!(data.traces || []).length && (
              <div className="empty">
                {auditMissing ? 'Auditoría no archivada para esta campaña.' : 'Selecciona una neurona para vigilar sus conexiones.'}
              </div>
            )}
          </div>
        </article>
      </div>
    </div>
  );
}
