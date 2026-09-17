import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Box, Rotate3d, SlidersHorizontal, Route, Download } from "lucide-react";
import {withRun} from "./Provenance.jsx";

const api = async (path) => {
  const response = await fetch("/api/" + path);
  if (!response.ok) throw Error(await response.text());
  return response.json();
};
const colors = {
  visual_projection: 0x2d80d0,
  sensory: 0xe96935,
  optic: 0x18a876,
  central: 0xf0ad00,
  vnc: 0xe66d9d,
  ascending: 0x08a526,
  descending: 0x5040aa,
  other: 0x8d98a2,
};
const labels = {
  visual_projection: "Proyección visual",
  sensory: "Sensorial",
  optic: "Óptico",
  central: "Central",
  vnc: "Cordón",
  ascending: "Ascendentes",
  descending: "Descendentes",
  other: "Otras",
};

export default function Brain3D({run,context}) {
  const mount = useRef(null),
    canvas = useRef(null),
    [limit, setLimit] = useState(() => window.matchMedia?.("(max-width: 720px)").matches ? 800 : 1400),
    [family, setFamily] = useState(""),
    [hemisphere, setHemisphere] = useState(""),
    [region, setRegion] = useState(""),
    [neuronType, setNeuronType] = useState(""),
    [activeOnly, setActiveOnly] = useState(false),
    [decisionOnly, setDecisionOnly] = useState(false),
    [cluster, setCluster] = useState(false),
    [edgeMode, setEdgeMode] = useState("synapses"),
    [data, setData] = useState(null),
    [causal, setCausal] = useState(null),
    [favorites, setFavorites] = useState(() => JSON.parse(localStorage.getItem("flybrain-favorites") || "[]")),
    [selected, setSelected] = useState(null),
    [error, setError] = useState(""),
    [renderError, setRenderError] = useState("");
  useEffect(() => {
    let active = true;
    const query = new URLSearchParams({ limit, family, hemisphere, region, neuron_type: neuronType, active_only: activeOnly, decision_only: decisionOnly });
    api(withRun(`brain/3d?${query}`,run))
      .then((next) => {
        if (active) {
          setData(next);
          setSelected(null);
          setError("");
          setRenderError("");
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [limit, family, hemisphere, region, neuronType, activeOnly, decisionOnly, run]);
  useEffect(() => { api(withRun("causal/current",run)).then(setCausal).catch(() => {}); }, [data,run]);
  useEffect(() => {
    if (!mount.current || !data) return;
    const host = mount.current,
      scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f8fa);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
    camera.position.set(0, 80, 260);
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "high-performance",
      });
    } catch (cause) {
      setRenderError(`No se pudo iniciar WebGL: ${cause.message || "contexto 3D no disponible"}`);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    canvas.current = renderer;
    host.replaceChildren(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.minDistance = 45;
    controls.maxDistance = 800;
    const coords = data.nodes.map(
      (node) => new THREE.Vector3(node.x, node.y, node.z),
    );
    const center = coords
      .reduce((sum, point) => sum.add(point), new THREE.Vector3())
      .multiplyScalar(1 / Math.max(1, coords.length));
    const extent = Math.max(
      ...coords.map((point) => point.distanceTo(center)),
      1,
    );
    const scale = 150 / extent;
    const positions = coords.map((point, index) => {
      const position = point.sub(center).multiplyScalar(scale);
      if (cluster) {
        const familyIndex = Object.keys(labels).indexOf(data.nodes[index].family);
        position.x += (familyIndex % 4 - 1.5) * 16;
        position.z += (Math.floor(familyIndex / 4) - 0.5) * 14;
      }
      return position;
    });
    const pointPositions = new Float32Array(positions.length * 3),
      pointColors = new Float32Array(positions.length * 3);
    positions.forEach((point, index) => {
      pointPositions.set([point.x, point.y, point.z], index * 3);
      const color = new THREE.Color(
        colors[data.nodes[index].family] || colors.other,
      );
      pointColors.set([color.r, color.g, color.b], index * 3);
    });
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(pointPositions, 3),
    );
    pointGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(pointColors, 3),
    );
    const points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({
        size: 2.8,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
      }),
    );
    scene.add(points);
    const activePositions = positions.filter(
      (_, index) => (data.nodes[index].spikes || 0) > 0,
    );
    if (activePositions.length) {
      const activeArray = new Float32Array(activePositions.length * 3);
      activePositions.forEach((point, index) =>
        activeArray.set([point.x, point.y, point.z], index * 3),
      );
      const activeGeometry = new THREE.BufferGeometry();
      activeGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(activeArray, 3),
      );
      scene.add(
        new THREE.Points(
          activeGeometry,
          new THREE.PointsMaterial({
            color: 0xfff2a8,
            size: 7,
            transparent: true,
            opacity: 0.75,
            sizeAttenuation: true,
          }),
        ),
      );
    }
    const lineArray = new Float32Array(data.edges.length * 6), lineColors = new Float32Array(data.edges.length * 6);
    data.edges.forEach((edge, index) => {
      const from = positions[edge.source],
        to = positions[edge.target];
      lineArray.set([from.x, from.y, from.z, to.x, to.y, to.z], index * 6);
      const color = new THREE.Color(edge.sign < 0 ? 0xd86b68 : 0x4e9dbe);
      if (edgeMode === "effective" && edge.gain != null) color.offsetHSL(0, 0, Math.max(-0.25, Math.min(0.25, (edge.gain - 1) * 0.08)));
      lineColors.set([color.r, color.g, color.b, color.r, color.g, color.b], index * 6);
    });
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(lineArray, 3),
    );
    lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
    scene.add(
      new THREE.LineSegments(
        lineGeometry,
        new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: 0.2,
        }),
      ),
    );
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const grid = new THREE.GridHelper(340, 12, 0xd8e2e8, 0xe9eff2);
    grid.position.y = -170;
    scene.add(grid);
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 5;
    const pointer = new THREE.Vector2();
    const click = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(points)[0];
      if (hit) setSelected(data.nodes[hit.index]);
    };
    renderer.domElement.addEventListener("pointerup", click);
    const resize = () => {
      camera.aspect = host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    const render = () => renderer.render(scene, camera);
    controls.addEventListener("change", render);
    render();
    return () => {
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerup", click);
      controls.removeEventListener("change", render);
      controls.dispose();
      pointGeometry.dispose();
      lineGeometry.dispose();
      renderer.dispose();
      canvas.current = null;
      host.replaceChildren();
    };
  }, [data, cluster, edgeMode]);
  useEffect(() => {
    if (data && !data.activity_available) {
      setActiveOnly(false);
      setEdgeMode("synapses");
    }
  }, [data?.activity_available]);
  const activityAvailable = Boolean(data?.activity_available && context?.live);
  const toggleFavorite = () => { if (!selected) return; const next = favorites.includes(selected.body_id) ? favorites.filter(id => id !== selected.body_id) : [...favorites, selected.body_id]; setFavorites(next); localStorage.setItem("flybrain-favorites", JSON.stringify(next)); };
  const capture = () => { if (!canvas.current) return; const link=document.createElement("a"); link.download=`flybrain-${data?.activity_revision||"snapshot"}.png`; link.href=canvas.current.domElement.toDataURL("image/png"); link.click(); };
  return (
    <article className="panel brain3d-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">CONECTOMA ANATÓMICO</span>
          <h2>Modelo cerebral 3D</h2>
          <p>
            Coordenadas reales de soma · color por familia · líneas = conexiones
            muestreadas del grafo publicado. {activityAvailable ? 'La actividad procede del snapshot actual.' : 'La actividad no está disponible para este contexto.'}
          </p>
        </div>
        <span className="tag">
          <Rotate3d size={14} /> Orbitar para explorar
        </span>
      </div>
      <div className="brain3d-tools">
        <label>
          <SlidersHorizontal size={14} /> Muestra
          <select
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
          >
            <option value="800">800 neuronas</option>
            <option value="1400">1,400 neuronas</option>
            <option value="2400">2,400 neuronas</option>
          </select>
        </label>
        <label>
          <Box size={14} /> Familia
          <select
            value={family}
            onChange={(event) => setFamily(event.target.value)}
          >
            <option value="">Todas</option>
            {Object.entries(labels).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <span>
          {data
            ? `${data.sampled_neurons.toLocaleString("es-MX")} neuronas · ${data.sampled_edges.toLocaleString("es-MX")} conexiones`
            : ""}
        </span>
        <label>Hemisferio <select value={hemisphere} onChange={event=>setHemisphere(event.target.value)}><option value="">Ambos</option><option value="left">Izquierdo</option><option value="right">Derecho</option></select></label>
        <label>Región <input value={region} onChange={event=>setRegion(event.target.value)} placeholder="neuromere" /></label>
        <label>Tipo <input value={neuronType} onChange={event=>setNeuronType(event.target.value)} placeholder="DNp01" /></label>
        <label><input type="checkbox" checked={activeOnly} disabled={!activityAvailable} onChange={event=>setActiveOnly(event.target.checked)} /> Solo activas</label>
        <label><input type="checkbox" checked={decisionOnly} onChange={event=>setDecisionOnly(event.target.checked)} /> Solo decisión</label>
        <label>Aristas <select value={edgeMode} onChange={event=>setEdgeMode(event.target.value)}><option value="synapses">Sinapsis</option><option value="effective" disabled={!activityAvailable}>Peso efectivo</option></select></label>
        <label><input type="checkbox" checked={cluster} onChange={event=>setCluster(event.target.checked)} /> Agrupar familias</label>
        <button className="btn" onClick={capture}><Download size={14}/> PNG 3D</button>
        <a className="btn" href={data ? '/api/'+withRun(`export/brain3d?limit=${limit}&family=${encodeURIComponent(family)}&hemisphere=${encodeURIComponent(hemisphere)}&region=${encodeURIComponent(region)}&neuron_type=${encodeURIComponent(neuronType)}&active_only=${activeOnly && activityAvailable}&decision_only=${decisionOnly}`,run) : undefined}><Download size={14}/> JSON</a><a className="btn" href={data ? '/api/'+withRun(`export/brain3d?format=csv&limit=${limit}&family=${encodeURIComponent(family)}&hemisphere=${encodeURIComponent(hemisphere)}&region=${encodeURIComponent(region)}&neuron_type=${encodeURIComponent(neuronType)}&active_only=${activeOnly && activityAvailable}&decision_only=${decisionOnly}`,run) : undefined}>CSV</a>
      </div>
      {error || renderError ? (
        <div className="brain3d-empty"><strong>No se pudo mostrar el cerebro 3D.</strong><span>{error || renderError}</span><small>Prueba habilitando aceleración gráfica/WebGL en el navegador.</small></div>
      ) : data && !data.nodes.length ? (
        <div className="brain3d-empty"><strong>No hay neuronas para estos filtros.</strong><span>Prueba quitando Región/Tipo, desactiva “Solo activas” o usa “Todas” en Familia.</span><small>{data.filters?.region ? `Región: ${data.filters.region}` : ""}{data.filters?.neuron_type ? ` · Tipo: ${data.filters.neuron_type}` : ""}{data.filters?.active_only ? " · Solo activas" : ""}{data.filters?.decision_only ? " · Solo decisión" : ""}</small></div>
      ) : (
        <div ref={mount} className="brain3d-canvas" />
      )}
      {selected && (
        <div className="brain3d-selection">
          <strong>
            {selected.type || "Neurona"} · bodyId {selected.body_id}
          </strong>
          <span>
            {labels[selected.family] || selected.family} · posición {selected.x}
            , {selected.y}, {selected.z} ·{" "}
            {selected.spikes == null
              ? "actividad no disponible"
              : `${selected.spikes} impulsos / 50 ms`}
          </span>
          <button className="btn" onClick={toggleFavorite}>{favorites.includes(selected.body_id)?"Quitar favorito":"Favorito"}</button><button className="btn" disabled={!context?.control_allowed || !activityAvailable} onClick={()=>api(withRun(`neuron/${selected.body_id}`,run)).then(()=>setSelected({...selected,watched:true}))}>Vigilar neurona</button>
          <button onClick={() => setSelected(null)}>Cerrar</button>
        </div>
      )}
      <div className="brain3d-legend">
        {Object.entries(labels).map(([key, label]) => (
          <span key={key}>
            <i
              style={{
                background: "#" + colors[key].toString(16).padStart(6, "0"),
              }}
            />
            {label}
          </span>
        ))}
        <span><i className="edge-excitatory" />Excitatoria</span>
        <span><i className="edge-inhibitory" />Inhibitoria</span>
        {activityAvailable&&<span><i className="active-dot" />Activa en la última ventana</span>}
      </div>
      <div className="causal-panel"><div className="panel-heading"><div><h3><Route size={16}/> {activityAvailable?'Ruta causal de la decisión actual':'Contribuciones archivadas de la última decisión'}</h3><p>{activityAvailable?<><b>Observado:</b> canales y neuronas activas. <b>Inferido:</b> contribución al logit del actor.</>:'La actividad no se reconstruye fuera del snapshot actual; las contribuciones que fueron persistidas siguen identificadas como históricas.'}</p></div><span className="tag">{causal?.inferred?.action||data?.last_decision?.action_label||'—'}</span></div>{activityAvailable?<><div className="causal-route"><span>Entrada sensorial</span><b>→</b><span>{causal?.observed?.active_readout?.length||0} neuronas activas</span><b>→</b><span>Descendentes</span><b>→</b><strong>{causal?.inferred?.action||'—'}</strong></div>{(causal?.inferred?.contributions||[]).slice(0,8).map(row=><div className="causal-row" key={row.body_id}><span>DN {row.body_id} · {row.filter_ms} ms</span><div><i style={{width:`${Math.min(100,Math.abs(row.logit_contribution)*1000)}%`,background:row.logit_contribution>=0?'#2b9b88':'#d86b68'}}/></div><strong>{Number(row.logit_contribution).toFixed(4)}</strong></div>)}</>:<><div className="circuit-archive"><strong>{data?.last_decision?.action_label||'No hay decisión archivada'}</strong><span>Se conserva la anatomía y la decisión durable, sin presentar actividad neuronal ni gains actuales.</span></div>{(causal?.inferred?.contributions||[]).slice(0,8).map(row=><div className="causal-row" key={row.body_id}><span>DN {row.body_id} · contribución archivada</span><div><i style={{width:`${Math.min(100,Math.abs(row.logit_contribution)*1000)}%`,background:row.logit_contribution>=0?'#2b9b88':'#d86b68'}}/></div><strong>{Number(row.logit_contribution).toFixed(4)}</strong></div>)}</>}</div>
      <p className="graph-note">
        La escena muestra una muestra limitada para conservar interacción. El
        conteo total del conectoma es{" "}
        {data?.total_neurons?.toLocaleString("es-MX") || "166,700"} neuronas y{" "}
        {data?.total_edges?.toLocaleString("es-MX") || "25,582,938"} aristas.
      </p>
    </article>
  );
}
