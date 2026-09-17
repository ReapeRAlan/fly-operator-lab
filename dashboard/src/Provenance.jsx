import React from "react";
import { Archive, CircleAlert, Radio, Database } from "lucide-react";

const labels = {
  live: "En vivo",
  "histórico": "Histórico",
  obsoleto: "Obsoleto",
  "sin datos": "Sin datos",
};

const icons = {
  live: Radio,
  "histórico": Archive,
  obsoleto: CircleAlert,
  "sin datos": Database,
};

export function withRun(path, run) {
  if (!run) return path;
  const [base, query = ""] = path.split("?");
  const params = new URLSearchParams(query);
  params.set("run", run);
  return `${base}?${params}`;
}

export function contextLabel(context) {
  return labels[context?.freshness] || "Sin datos";
}

export function Provenance({ context, compact = false }) {
  const kind = context?.freshness || "sin datos";
  const Icon = icons[kind] || Database;
  if (!context) return null;
  return (
    <aside className={`provenance provenance-${kind}${compact ? " compact" : ""}`} aria-live="polite">
      <Icon size={compact ? 14 : 17} />
      <div>
        <strong>{contextLabel(context)} · {context.selected_run}</strong>
        {!compact && <span>{context.reason}</span>}
      </div>
      {!compact && context.source_run && context.source_run !== context.selected_run && (
        <small>snapshot detectado: {context.source_run}</small>
      )}
    </aside>
  );
}
