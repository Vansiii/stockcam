"""StockCam: inventario de productos con la cámara del celular.

El navegador detecta y clasifica (TensorFlow.js); este servidor guarda
productos, movimientos de stock y las muestras de entrenamiento en SQLite,
así el celular y la PC ven el mismo inventario.
"""
import argparse
import json
import math
import os
import socket
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, abort, g, jsonify, request
from werkzeug.exceptions import HTTPException

BASE = Path(__file__).parent
DIAS_CONSUMO = 14      # ventana para calcular el consumo diario
DIAS_OBJETIVO = 7      # el pedido sugerido cubre una semana más el mínimo

app = Flask(__name__, static_folder="static", static_url_path="")
# En Railway, RAILWAY_VOLUME_MOUNT_PATH apunta al volumen persistente.
_datos = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", BASE))
app.config["DB"] = os.environ.get("STOCKCAM_DB", str(_datos / "inventario.db"))

ESQUEMA = """
CREATE TABLE IF NOT EXISTS productos(
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL UNIQUE,
    codigo TEXT UNIQUE,
    stock_min INTEGER NOT NULL DEFAULT 0 CHECK (stock_min >= 0),
    creado TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS movimientos(
    id INTEGER PRIMARY KEY,
    producto_id INTEGER NOT NULL REFERENCES productos(id) ON DELETE CASCADE,
    cantidad INTEGER NOT NULL CHECK (cantidad <> 0),
    tipo TEXT NOT NULL CHECK (tipo IN ('inicial', 'entrada', 'salida', 'ajuste')),
    fecha TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS muestras(
    id INTEGER PRIMARY KEY,
    producto_id INTEGER REFERENCES productos(id) ON DELETE CASCADE,  -- NULL = fondo (sin producto)
    vector TEXT NOT NULL
);
"""


def conectar():
    con = sqlite3.connect(app.config["DB"])
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def db():
    if "db" not in g:
        g.db = conectar()
    return g.db


@app.teardown_appcontext
def cerrar(_):
    con = g.pop("db", None)
    if con is not None:
        con.close()


def iniciar_db():
    with conectar() as con:
        con.executescript(ESQUEMA)
        # La primera versión no admitía muestras de fondo (producto_id NOT NULL): se migra.
        if any(c["name"] == "producto_id" and c["notnull"] for c in con.execute("PRAGMA table_info(muestras)")):
            con.executescript("ALTER TABLE muestras RENAME TO muestras_v1;" + ESQUEMA +
                              "INSERT INTO muestras SELECT * FROM muestras_v1; DROP TABLE muestras_v1;")


@app.errorhandler(HTTPException)
def error_http(e):
    return jsonify(error=e.description), e.code


@app.errorhandler(sqlite3.IntegrityError)
def error_duplicado(_):
    return jsonify(error="Ya existe un producto con ese nombre o código"), 409


# ---------- validación de entrada ----------

def cuerpo():
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        abort(400, "Se esperaba un JSON")
    return datos


def entero(valor, campo, minimo=None):
    if isinstance(valor, bool) or not isinstance(valor, (int, str)):
        abort(400, f"{campo} debe ser un número entero")
    try:
        n = int(valor)
    except ValueError:
        abort(400, f"{campo} debe ser un número entero")
    if minimo is not None and n < minimo:
        abort(400, f"{campo} debe ser al menos {minimo}")
    return n


def texto(valor, campo, obligatorio=True):
    valor = (valor or "").strip() if isinstance(valor, str) or valor is None else None
    if valor is None or len(valor) > 80:
        abort(400, f"{campo} no es válido")
    if obligatorio and not valor:
        abort(400, f"{campo} es obligatorio")
    return valor or None


def stock_de(pid):
    fila = db().execute(
        "SELECT COALESCE(SUM(cantidad), 0) FROM movimientos WHERE producto_id = ?", (pid,)
    ).fetchone()
    return fila[0]


def producto_o_404(pid):
    if not db().execute("SELECT 1 FROM productos WHERE id = ?", (pid,)).fetchone():
        abort(404, "Producto no encontrado")


# ---------- rutas ----------

@app.get("/")
def inicio():
    return app.send_static_file("index.html")


@app.get("/api/productos")
def listar_productos():
    filas = db().execute("""
        SELECT p.id, p.nombre, p.codigo, p.stock_min,
               COALESCE((SELECT SUM(cantidad) FROM movimientos WHERE producto_id = p.id), 0) AS stock,
               (SELECT COUNT(*) FROM muestras WHERE producto_id = p.id) AS muestras
        FROM productos p ORDER BY p.nombre COLLATE NOCASE
    """).fetchall()
    return jsonify([dict(f) for f in filas])


@app.post("/api/productos")
def crear_producto():
    d = cuerpo()
    nombre = texto(d.get("nombre"), "El nombre")
    codigo = texto(d.get("codigo"), "El código", obligatorio=False)
    stock_min = entero(d.get("stock_min", 0), "El stock mínimo", 0)
    inicial = entero(d.get("stock_inicial", 0), "El stock inicial", 0)
    con = db()
    with con:
        pid = con.execute(
            "INSERT INTO productos(nombre, codigo, stock_min) VALUES (?, ?, ?)",
            (nombre, codigo, stock_min),
        ).lastrowid
        if inicial:
            con.execute(
                "INSERT INTO movimientos(producto_id, cantidad, tipo) VALUES (?, ?, 'inicial')",
                (pid, inicial),
            )
    return jsonify(id=pid), 201


@app.patch("/api/productos/<int:pid>")
def editar_producto(pid):
    producto_o_404(pid)
    d = cuerpo()
    cambios = {}
    if "nombre" in d:
        cambios["nombre"] = texto(d["nombre"], "El nombre")
    if "codigo" in d:
        cambios["codigo"] = texto(d["codigo"], "El código", obligatorio=False)
    if "stock_min" in d:
        cambios["stock_min"] = entero(d["stock_min"], "El stock mínimo", 0)
    if cambios:
        columnas = ", ".join(f"{c} = ?" for c in cambios)  # claves fijas, no vienen del usuario
        with db() as con:
            con.execute(f"UPDATE productos SET {columnas} WHERE id = ?", (*cambios.values(), pid))
    return jsonify(ok=True)


@app.delete("/api/productos/<int:pid>")
def borrar_producto(pid):
    producto_o_404(pid)
    with db() as con:
        con.execute("DELETE FROM productos WHERE id = ?", (pid,))
    return jsonify(ok=True)


@app.post("/api/productos/<int:pid>/movimientos")
def mover_stock(pid):
    """{"cantidad": +n | -n} suma o resta; {"conteo": n} fija el stock contado."""
    producto_o_404(pid)
    d = cuerpo()
    stock = stock_de(pid)
    if "conteo" in d:
        cantidad = entero(d["conteo"], "El conteo", 0) - stock
        tipo = "ajuste"
    else:
        cantidad = entero(d.get("cantidad"), "La cantidad")
        tipo = "entrada" if cantidad > 0 else "salida"
        if stock + cantidad < 0:
            abort(409, f"Solo hay {stock} en inventario")
    if cantidad:
        with db() as con:
            con.execute(
                "INSERT INTO movimientos(producto_id, cantidad, tipo) VALUES (?, ?, ?)",
                (pid, cantidad, tipo),
            )
    return jsonify(stock=stock + cantidad)


@app.get("/api/muestras")
def listar_muestras():
    filas = db().execute("SELECT producto_id, vector FROM muestras").fetchall()
    return jsonify([{"producto_id": f["producto_id"], "vector": json.loads(f["vector"])} for f in filas])


@app.post("/api/productos/<int:pid>/muestras")
def guardar_muestra(pid):
    producto_o_404(pid)
    return insertar_muestra(pid)


# El fondo es la escena sin productos: el clasificador lo usa para decir "aquí no hay nada".
@app.post("/api/fondo/muestras")
def guardar_fondo():
    return insertar_muestra(None)


@app.delete("/api/fondo/muestras")
def borrar_fondo():
    with db() as con:
        con.execute("DELETE FROM muestras WHERE producto_id IS NULL")
    return jsonify(ok=True)


def insertar_muestra(pid):
    v = cuerpo().get("vector")
    valido = (
        isinstance(v, list) and 64 <= len(v) <= 2048
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)
    )
    if not valido:
        abort(400, "Vector de muestra no válido")
    with db() as con:
        con.execute("INSERT INTO muestras(producto_id, vector) VALUES (?, ?)", (pid, json.dumps(v)))
    return jsonify(ok=True), 201


@app.delete("/api/productos/<int:pid>/muestras")
def borrar_muestras(pid):
    producto_o_404(pid)
    with db() as con:
        con.execute("DELETE FROM muestras WHERE producto_id = ?", (pid,))
    return jsonify(ok=True)


@app.get("/api/resumen")
def resumen():
    """Indicadores para decidir qué reponer: consumo, cobertura y pedido sugerido."""
    desde = (datetime.now(timezone.utc) - timedelta(days=DIAS_CONSUMO)).strftime("%Y-%m-%d %H:%M:%S")
    filas = db().execute("""
        SELECT p.id, p.nombre, p.stock_min,
               COALESCE(SUM(m.cantidad), 0) AS stock,
               COALESCE(-SUM(CASE WHEN m.tipo = 'salida' AND m.fecha >= ? THEN m.cantidad END), 0) AS salidas
        FROM productos p LEFT JOIN movimientos m ON m.producto_id = p.id
        GROUP BY p.id ORDER BY p.nombre COLLATE NOCASE
    """, (desde,)).fetchall()

    productos = []
    for f in filas:
        consumo = f["salidas"] / DIAS_CONSUMO
        stock = f["stock"]
        productos.append({
            "id": f["id"],
            "nombre": f["nombre"],
            "stock": stock,
            "stock_min": f["stock_min"],
            "consumo_diario": round(consumo, 2),
            "cobertura_dias": round(stock / consumo, 1) if consumo else None,
            "pedir": max(0, math.ceil(f["stock_min"] + consumo * DIAS_OBJETIVO - stock)),
            "estado": "agotado" if stock <= 0 else "bajo" if stock <= f["stock_min"] else "ok",
        })

    movimientos = db().execute("""
        SELECT m.cantidad, m.tipo, m.fecha, p.nombre
        FROM movimientos m JOIN productos p ON p.id = m.producto_id
        ORDER BY m.id DESC LIMIT 20
    """).fetchall()
    return jsonify(productos=productos, movimientos=[dict(m) for m in movimientos])


def ip_local():
    # No envía nada: solo pregunta al sistema qué interfaz usaría para salir a la red.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


iniciar_db()  # idempotente; también corre cuando gunicorn importa el módulo

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="StockCam")
    p.add_argument("--https", action="store_true",
                   help="certificado autofirmado; el celular solo abre la cámara con https")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    esquema = "https" if args.https else "http"
    print(f"\n  En esta PC:   {esquema}://127.0.0.1:{args.port}")
    print(f"  En el celular: {esquema}://{ip_local()}:{args.port}  (misma red Wi-Fi)\n")
    app.run(host="0.0.0.0", port=args.port, ssl_context="adhoc" if args.https else None)
