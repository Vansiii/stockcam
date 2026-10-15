"use strict";

// Clases de COCO-SSD que tratamos como "producto". Personas, autos, muebles, etc. se ignoran.
const CLASES = {
  bottle: "botella", cup: "vaso", "wine glass": "copa", bowl: "tazón", banana: "plátano",
  apple: "manzana", orange: "naranja", broccoli: "brócoli", carrot: "zanahoria",
  sandwich: "sándwich", donut: "dona", cake: "pastel", book: "libro", "cell phone": "celular",
  remote: "control", scissors: "tijeras", toothbrush: "cepillo", vase: "florero",
  "sports ball": "pelota", "teddy bear": "peluche", mouse: "mouse", clock: "reloj",
};
const K = 5;           // vecinos que votan en el clasificador
const VENTANA = 10;    // cuadros usados para estabilizar el conteo
const LADO = 224;      // entrada de MobileNet
const ESPERA_AUTO = 2000;  // ms que el conteo debe mantenerse igual para fijarlo solo

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const video = $("#video");
const lienzo = $("#lienzo");
const ctx = lienzo.getContext("2d");

const st = {
  productos: [],
  muestras: [],          // [{id, v: Float32Array normalizado}]
  coco: null, net: null, stream: null,
  modo: "contar",
  capturar: 0, cuadro: 0,
  historial: [],         // conteos de los últimos cuadros
  estables: new Map(),   // id → {n, desde, hecho} para el conteo automático
  conteoMostrado: "",
  exigencia: Number(leer("exigencia") ?? 0.6),
};

function leer(k) { try { return localStorage.getItem("stockcam-" + k); } catch { return null; } }
function guardar(k, v) { try { localStorage.setItem("stockcam-" + k, v); } catch { /* sin almacenamiento */ } }

async function api(ruta, opciones = {}) {
  const r = await fetch(ruta, {
    ...opciones,
    headers: { "Content-Type": "application/json" },
    body: opciones.body && JSON.stringify(opciones.body),
  });
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(datos.error || `Error ${r.status}`);
  return datos;
}

let temporizador;
function aviso(texto, error = false) {
  const el = $("#aviso");
  el.textContent = texto;
  el.className = "aviso" + (error ? " error" : "");
  el.hidden = false;
  clearTimeout(temporizador);
  temporizador = setTimeout(() => (el.hidden = true), 3000);
}

const producto = (id) => st.productos.find((p) => p.id === id);
const color = (id) => `hsl(${(id * 137) % 360} 80% 45%)`;

// ---------- navegación ----------

function mostrar(vista) {
  document.querySelectorAll(".vista").forEach((v) => v.classList.toggle("activa", v.id === "v-" + vista));
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("activa", b.dataset.vista === vista));
  if (vista !== "escanear") apagarCamara();
  if (vista === "panel") cargarPanel();
}
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => mostrar(b.dataset.vista)));

// ---------- datos ----------

async function cargarProductos() {
  st.productos = await api("/api/productos");
  const unidades = st.productos.reduce((s, p) => s + p.stock, 0);
  $("#total").textContent = `${unidades} unidades · ${st.productos.length} productos`;
  pintarProductos();
  pintarModos();
  st.conteoMostrado = "";   // fuerza a repintar el conteo con el stock nuevo
}

async function cargarMuestras() {
  const filas = await api("/api/muestras");
  st.muestras = filas.map((m) => ({ id: m.producto_id, v: normalizar(m.vector) }));
}

function normalizar(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}

// ---------- clasificador: k vecinos más cercanos sobre embeddings de MobileNet ----------

function clasificar(vector) {
  if (!st.muestras.length) return null;
  const e = normalizar(vector);
  const cercanos = st.muestras
    .map((m) => { let s = 0; for (let i = 0; i < e.length; i++) s += e[i] * m.v[i]; return [s, m.id]; })
    .sort((a, b) => b[0] - a[0])
    .slice(0, K);
  const votos = new Map();
  for (const [s, id] of cercanos) votos.set(id, (votos.get(id) || 0) + s);
  const [id, suma] = [...votos].sort((a, b) => b[1] - a[1])[0];
  // Siempre devuelve el más parecido; quien llama decide con la exigencia (así se ve el % al calibrar).
  return { id, parecido: suma / cercanos.filter(([, i]) => i === id).length };
}

// Recorta todas las cajas del cuadro actual y saca sus embeddings en un solo lote.
async function vectores(cajas) {
  const t = tf.tidy(() => {
    const img = tf.browser.fromPixels(video);
    const [H, W] = img.shape;
    const normal = cajas.map(([x, y, w, h]) => [y / H, x / W, (y + h) / H, (x + w) / W]);
    const recortes = tf.image.cropAndResize(img.expandDims(0).toFloat(), normal, cajas.map(() => 0), [LADO, LADO]);
    return st.net.infer(recortes, true);
  });
  const datos = await t.array();
  t.dispose();
  return datos;
}

// En modo entrenar se usa el objeto más grande; si no hay, el centro de la imagen.
function cajaEntrenamiento(dets) {
  if (dets.length) return dets.reduce((a, b) => (a.bbox[2] * a.bbox[3] > b.bbox[2] * b.bbox[3] ? a : b)).bbox;
  const W = video.videoWidth, H = video.videoHeight, l = Math.min(W, H) * 0.6;
  return [(W - l) / 2, (H - l) / 2, l, l];
}

// ---------- cámara ----------

async function encenderCamara() {
  if (!window.isSecureContext) {
    $("#estado-camara").textContent = "El navegador solo permite la cámara con https. Inicia el servidor con: python app.py --https";
    return;
  }
  const boton = $("#b-camara");
  boton.disabled = true;
  try {
    $("#estado-camara").textContent = "Cargando modelos (la primera vez tarda unos segundos)…";
    [st.coco, st.net] = await Promise.all([
      st.coco || cocoSsd.load({ base: "lite_mobilenet_v2" }),
      st.net || mobilenet.load({ version: 2, alpha: 1 }),
    ]);
    st.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false,
    });
    video.srcObject = st.stream;
    await video.play();
    lienzo.width = video.videoWidth;
    lienzo.height = video.videoHeight;
    $("#tapa").hidden = true;
    $("#estado-camara").textContent = !st.muestras.length
      ? "Aún no hay productos entrenados: se mostrará el tipo de objeto. Entrena uno desde «Modo»."
      : !muestrasFondo()
        ? "Consejo: entrena también el «fondo» (Modo → Entrenar: fondo) para que no confunda la escena con un producto."
        : "";
    ciclo();
  } catch (e) {
    $("#estado-camara").textContent = "No se pudo abrir la cámara: " + e.message;
  } finally {
    boton.disabled = false;
  }
}

function apagarCamara() {
  st.stream?.getTracks().forEach((t) => t.stop());
  st.stream = null;
  $("#tapa").hidden = false;
  ctx.clearRect(0, 0, lienzo.width, lienzo.height);
}

// El video se reproduce solo, sin esperar al modelo: por eso no hay lag.
// La detección corre en paralelo y solo actualiza las cajas encima.
async function ciclo() {
  if (!st.stream) return;
  try {
    const dets = (await st.coco.detect(video, 20, 0.3)).filter((d) => d.class in CLASES);
    const entrenando = st.modo !== "contar";
    // Igual que al entrenar: si el detector no encuentra nada, se clasifica el centro de la imagen.
    // Así también se cuentan productos que COCO-SSD no conoce (cajas, bolsas…), de uno en uno.
    const centro = !entrenando && !dets.length && st.muestras.length > 0;
    const cajas = entrenando || centro ? [cajaEntrenamiento(dets)] : dets.map((d) => d.bbox);
    const vecs = cajas.length ? await vectores(cajas) : [];
    if (!st.stream) return;

    if (entrenando) {
      dibujar([{ caja: cajas[0], etiqueta: "muestra", tono: "hsl(45 100% 50%)" }]);
      if (st.capturar > 0 && st.cuadro % 3 === 0) await capturar(st.modo === "fondo" ? null : Number(st.modo), vecs[0]);
    } else {
      const items = cajas.map((caja, i) => {
        const r = clasificar(vecs[i]);
        const pct = r ? ` ${Math.round(r.parecido * 100)}%` : "";
        if (r && r.id !== null && r.parecido >= st.exigencia) {
          return { caja, clave: r.id, etiqueta: producto(r.id)?.nombre + pct, tono: color(r.id) };
        }
        // Lo no reconocido se dibuja con su % para poder calibrar la exigencia, pero no se cuenta.
        // Si gana el fondo, lo que hay en el centro es solo escena: no se dibuja ni se cuenta.
        if (centro && r?.id === null) return null;
        const nombre = centro ? "centro" : CLASES[dets[i].class];
        return { caja, clave: centro ? null : "?" + dets[i].class, etiqueta: `${nombre} ?${pct}`, tono: "hsl(0 0% 60%)" };
      }).filter(Boolean);
      dibujar(items);
      contar(items.filter((it) => it.clave !== null));
    }
    if (st.cuadro % 10 === 0) await leerCodigo();
    st.cuadro++;
  } catch (e) {
    console.error(e);
  }
  requestAnimationFrame(ciclo);
}

function dibujar(items) {
  ctx.clearRect(0, 0, lienzo.width, lienzo.height);
  const grosor = Math.max(2, lienzo.width / 300);
  const letra = Math.round(Math.max(14, lienzo.width / 40));
  const alto = letra + 8;
  ctx.font = `600 ${letra}px system-ui, sans-serif`;
  ctx.lineWidth = grosor;
  for (const { caja: [x, y, w, h], etiqueta, tono } of items) {
    ctx.strokeStyle = ctx.fillStyle = tono;
    ctx.strokeRect(x, y, w, h);
    const ancho = ctx.measureText(etiqueta).width + 12;
    const ty = y > alto ? y - alto : y;
    ctx.fillRect(x - grosor / 2, ty, ancho, alto);
    ctx.fillStyle = "#fff";
    ctx.fillText(etiqueta, x + 6, ty + alto - 7);
  }
}

// ---------- conteo estable ----------

function contar(items) {
  const actual = new Map();
  for (const it of items) actual.set(it.clave, (actual.get(it.clave) || 0) + 1);
  st.historial.push(actual);
  if (st.historial.length > VENTANA) st.historial.shift();

  // Mediana por clave: un cuadro con una detección de más o de menos no mueve el número.
  const claves = new Set(st.historial.flatMap((m) => [...m.keys()]));
  const estable = [];
  for (const k of claves) {
    const valores = st.historial.map((m) => m.get(k) || 0).sort((a, b) => a - b);
    const n = valores[Math.floor(valores.length / 2)];
    if (n) estable.push([k, n]);
  }
  const firma = JSON.stringify(estable);
  if (firma !== st.conteoMostrado) {   // repintar solo si cambió, para no romper los clics
    st.conteoMostrado = firma;
    pintarConteo(estable);
  }
  autoConteo(estable);
}

// Fija el inventario de cada producto reconocido cuando su conteo lleva ESPERA_AUTO sin cambiar.
// Un producto que sale de cuadro no se toca: apartar la cámara no debe dejar el stock en 0.
function autoConteo(estable) {
  if (!$("#auto").checked) return;
  const ahora = performance.now();
  for (const [id, n] of estable) {
    if (typeof id !== "number") continue;
    const e = st.estables.get(id);
    if (!e || e.n !== n) {
      st.estables.set(id, { n, desde: ahora, hecho: false });
    } else if (!e.hecho && ahora - e.desde >= ESPERA_AUTO) {
      e.hecho = true;
      if (producto(id)?.stock !== n) fijar(id, n, true);
    }
  }
}

async function fijar(id, n, automatico = false) {
  try {
    await api(`/api/productos/${id}/movimientos`, { method: "POST", body: { conteo: n } });
    aviso(`${producto(id).nombre}: inventario fijado en ${n}${automatico ? " (automático)" : ""}`);
    await cargarProductos();
  } catch (err) { aviso(err.message, true); }
}

function pintarConteo(estable) {
  const ul = $("#conteo");
  if (!estable.length) {
    ul.innerHTML = `<li class="vacio">No veo productos todavía.</li>`;
    return;
  }
  ul.innerHTML = estable.map(([k, n]) => {
    const p = typeof k === "number" && producto(k);
    if (!p) {
      return `<li class="item"><span class="punto" style="background:hsl(0 0% 60%)"></span>
        <div class="crece"><strong>${n} × ${esc(CLASES[k.slice(1)])}</strong>
        <small>No reconocido. Créalo en Productos y entrénalo.</small></div></li>`;
    }
    return `<li class="item"><span class="punto" style="background:${color(p.id)}"></span>
      <div class="crece"><strong>${esc(p.nombre)}</strong>
      <small>En cámara: <b>${n}</b> · En inventario: <b>${p.stock}</b></small></div>
      <div class="acciones">
        <button class="btn" data-id="${p.id}" data-cantidad="${n}">+${n} entrada</button>
        <button class="btn" data-id="${p.id}" data-cantidad="${-n}">−${n} salida</button>
        <button class="btn" data-id="${p.id}" data-conteo="${n}" title="Hice el conteo físico: el inventario es ${n}">Fijar ${n}</button>
      </div></li>`;
  }).join("");
}

$("#conteo").onclick = async (e) => {
  const b = e.target.closest("button[data-id]");
  if (!b) return;
  const id = Number(b.dataset.id);
  if ("conteo" in b.dataset) return fijar(id, Number(b.dataset.conteo));
  try {
    const r = await api(`/api/productos/${id}/movimientos`, { method: "POST", body: { cantidad: Number(b.dataset.cantidad) } });
    aviso(`${producto(id).nombre}: ahora hay ${r.stock}`);
    await cargarProductos();
  } catch (err) { aviso(err.message, true); }
};

// ---------- entrenamiento ----------

function pintarModos() {
  const sel = $("#modo");
  const actual = sel.value;
  sel.innerHTML = `<option value="contar">Contar productos</option>` +
    st.productos.map((p) => `<option value="${p.id}">Entrenar: ${esc(p.nombre)}</option>`).join("") +
    `<option value="fondo">Entrenar: fondo (escena sin productos)</option>`;
  sel.value = [...sel.options].some((o) => o.value === actual) ? actual : "contar";
  cambiarModo();
}

const muestrasFondo = () => st.muestras.filter((m) => m.id === null).length;

function cambiarModo() {
  st.modo = $("#modo").value;
  const p = producto(Number(st.modo));
  const fondo = st.modo === "fondo";
  $("#panel-entrenar").hidden = !p && !fondo;
  $("#conteo").hidden = !!p || fondo;
  if (p) {
    $("#nombre-entrenar").textContent = p.nombre;
    $("#cuenta-muestras").textContent = `${p.muestras} muestras guardadas`;
  } else if (fondo) {
    $("#nombre-entrenar").textContent = "la escena SIN productos (tu mesa, la pared, tú mismo con las manos vacías)";
    $("#cuenta-muestras").textContent = `${muestrasFondo()} muestras guardadas`;
  }
  st.historial = [];
  st.estables.clear();
}
$("#modo").onchange = cambiarModo;

$("#b-capturar").onclick = () => {
  if (!st.stream) return aviso("Activa la cámara primero", true);
  st.capturar = 10;
};

async function capturar(id, vector) {
  const v = vector.map((x) => Math.round(x * 1e4) / 1e4);   // menos bytes, misma precisión útil
  await api(id === null ? "/api/fondo/muestras" : `/api/productos/${id}/muestras`, { method: "POST", body: { vector: v } });
  st.muestras.push({ id, v: normalizar(v) });
  const p = producto(id);
  if (p) p.muestras++;
  const total = p ? p.muestras : muestrasFondo();
  $("#cuenta-muestras").textContent = `${total} muestras guardadas`;
  if (--st.capturar === 0) aviso(`Listo: ${total} muestras de ${p ? p.nombre : "fondo"}`);
}

// ---------- código de barras (Chrome en Android lo trae de serie) ----------

const lector = "BarcodeDetector" in window ? new BarcodeDetector() : null;

async function leerCodigo() {
  if (!lector) return;
  const [c] = await lector.detect(video);
  const caja = $("#codigo");
  if (!c) return;
  const p = st.productos.find((x) => x.codigo === c.rawValue);
  caja.hidden = false;
  caja.innerHTML = p
    ? `Código <b>${esc(c.rawValue)}</b>: <strong>${esc(p.nombre)}</strong> · ${p.stock} en inventario`
    : `Código <b>${esc(c.rawValue)}</b> no registrado. <button class="btn" id="b-registrar">Registrar</button>`;
  $("#b-registrar")?.addEventListener("click", () => {
    mostrar("productos");
    const f = $("#form-producto");
    f.codigo.value = c.rawValue;
    f.nombre.focus();
  });
}

// ---------- productos ----------

const ESTADOS = { ok: "OK", bajo: "Bajo", agotado: "Agotado" };
const estadoDe = (p) => (p.stock <= 0 ? "agotado" : p.stock <= p.stock_min ? "bajo" : "ok");

function pintarProductos() {
  const ul = $("#productos");
  if (!st.productos.length) {
    ul.innerHTML = `<li class="vacio">Todavía no hay productos. Crea el primero arriba.</li>`;
    return;
  }
  ul.innerHTML = st.productos.map((p) => {
    const e = estadoDe(p);
    return `<li class="item producto" data-id="${p.id}">
      <span class="punto" style="background:${color(p.id)}"></span>
      <div class="crece">
        <strong>${esc(p.nombre)}</strong> <span class="chip ${e}">${ESTADOS[e]}</span>
        <small>${p.codigo ? "Código " + esc(p.codigo) + " · " : ""}${p.muestras} muestras ·
          mínimo <input class="min" type="number" min="0" value="${p.stock_min}" aria-label="Stock mínimo"></small>
      </div>
      <div class="stock">
        <button class="btn redondo" data-accion="restar" aria-label="Quitar uno">−</button>
        <b>${p.stock}</b>
        <button class="btn redondo" data-accion="sumar" aria-label="Agregar uno">+</button>
      </div>
      <div class="acciones">
        <button class="btn" data-accion="entrenar">Entrenar</button>
        ${p.muestras ? `<button class="btn" data-accion="olvidar">Borrar muestras</button>` : ""}
        <button class="btn peligro" data-accion="borrar">Eliminar</button>
      </div>
    </li>`;
  }).join("");
}

$("#productos").onclick = async (e) => {
  const b = e.target.closest("button[data-accion]");
  if (!b) return;
  const id = Number(b.closest("[data-id]").dataset.id);
  const p = producto(id);
  try {
    switch (b.dataset.accion) {
      case "sumar":
      case "restar":
        await api(`/api/productos/${id}/movimientos`, { method: "POST", body: { cantidad: b.dataset.accion === "sumar" ? 1 : -1 } });
        break;
      case "entrenar":
        mostrar("escanear");
        $("#modo").value = id;
        cambiarModo();
        return;
      case "olvidar":
        if (!confirm(`¿Borrar las ${p.muestras} muestras de ${p.nombre}?`)) return;
        await api(`/api/productos/${id}/muestras`, { method: "DELETE" });
        await cargarMuestras();
        break;
      case "borrar":
        if (!confirm(`¿Eliminar ${p.nombre} con todo su historial?`)) return;
        await api(`/api/productos/${id}`, { method: "DELETE" });
        await cargarMuestras();
        break;
    }
    await cargarProductos();
  } catch (err) { aviso(err.message, true); }
};

$("#productos").onchange = async (e) => {
  if (!e.target.matches("input.min")) return;
  const id = e.target.closest("[data-id]").dataset.id;
  try {
    await api(`/api/productos/${id}`, { method: "PATCH", body: { stock_min: Number(e.target.value) } });
    await cargarProductos();
  } catch (err) { aviso(err.message, true); }
};

$("#form-producto").onsubmit = async (e) => {
  e.preventDefault();
  const f = e.target;
  const datos = Object.fromEntries(new FormData(f));
  try {
    await api("/api/productos", { method: "POST", body: datos });
    aviso(`${datos.nombre} creado. Ahora entrénalo con la cámara.`);
    f.reset();
    await cargarProductos();
  } catch (err) { aviso(err.message, true); }
};

// ---------- panel ----------

async function cargarPanel() {
  try {
    const { productos, movimientos } = await api("/api/resumen");
    const cuenta = (e) => productos.filter((p) => p.estado === e).length;
    const kpis = [
      ["Unidades", productos.reduce((s, p) => s + p.stock, 0)],
      ["Productos", productos.length],
      ["Bajo mínimo", cuenta("bajo")],
      ["Agotados", cuenta("agotado")],
    ];
    $("#kpis").innerHTML = kpis.map(([t, v]) => `<div class="kpi"><small>${t}</small><b>${v}</b></div>`).join("");

    const tope = Math.max(1, ...productos.map((p) => Math.max(p.stock, p.stock_min)));
    $("#tabla-resumen").innerHTML = productos.length
      ? `<thead><tr><th>Producto</th><th>Stock</th><th>Consumo/día</th><th>Cobertura</th><th>Pedir</th></tr></thead><tbody>` +
        productos.map((p) => `<tr>
          <td>${esc(p.nombre)} <span class="chip ${p.estado}">${ESTADOS[p.estado]}</span>
            <div class="barra-stock"><i style="width:${(p.stock / tope) * 100}%;background:${color(p.id)}"></i>
            <em style="left:${(p.stock_min / tope) * 100}%" title="Mínimo ${p.stock_min}"></em></div></td>
          <td>${p.stock}</td>
          <td>${p.consumo_diario}</td>
          <td>${p.cobertura_dias == null ? "—" : p.cobertura_dias + " días"}</td>
          <td><b>${p.pedir || "—"}</b></td></tr>`).join("") + "</tbody>"
      : `<tbody><tr><td>Sin productos.</td></tr></tbody>`;

    const TIPOS = { inicial: "Stock inicial", entrada: "Entrada", salida: "Salida", ajuste: "Ajuste por conteo" };
    $("#movimientos").innerHTML = movimientos.length
      ? movimientos.map((m) => `<li><span class="${m.cantidad > 0 ? "mas" : "menos"}">${m.cantidad > 0 ? "+" : ""}${m.cantidad}</span>
          ${esc(m.nombre)} <small>${TIPOS[m.tipo]} · ${new Date(m.fecha.replace(" ", "T") + "Z").toLocaleString()}</small></li>`).join("")
      : `<li class="vacio">Sin movimientos.</li>`;
  } catch (err) { aviso(err.message, true); }
}

// ---------- inicio ----------

$("#b-camara").onclick = encenderCamara;
const auto = $("#auto");
auto.checked = leer("auto") === "1";
auto.onchange = () => { st.estables.clear(); guardar("auto", auto.checked ? "1" : "0"); };
const rango = $("#exigencia");
rango.value = st.exigencia;
$("#v-exigencia").textContent = Math.round(st.exigencia * 100) + "%";
rango.oninput = () => {
  st.exigencia = Number(rango.value);
  $("#v-exigencia").textContent = Math.round(st.exigencia * 100) + "%";
  guardar("exigencia", rango.value);
};

Promise.all([cargarProductos(), cargarMuestras()]).catch((e) => aviso(e.message, true));
