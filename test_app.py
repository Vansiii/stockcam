"""Prueba rápida de la API: python test_app.py"""
import os
import tempfile

import app as servidor


def main():
    with tempfile.TemporaryDirectory() as tmp:
        servidor.app.config["DB"] = os.path.join(tmp, "prueba.db")
        servidor.iniciar_db()
        c = servidor.app.test_client()

        r = c.post("/api/productos", json={"nombre": "Agua 600 ml", "codigo": "7750", "stock_min": 5, "stock_inicial": 10})
        assert r.status_code == 201, r.json
        pid = r.json["id"]

        assert c.post("/api/productos", json={"nombre": "Agua 600 ml"}).status_code == 409
        assert c.post("/api/productos", json={"nombre": ""}).status_code == 400
        assert c.post("/api/productos", json={"nombre": "X", "stock_min": -1}).status_code == 400

        assert c.post(f"/api/productos/{pid}/movimientos", json={"cantidad": 3}).json["stock"] == 13
        assert c.post(f"/api/productos/{pid}/movimientos", json={"cantidad": -7}).json["stock"] == 6
        assert c.post(f"/api/productos/{pid}/movimientos", json={"cantidad": -99}).status_code == 409
        assert c.post(f"/api/productos/{pid}/movimientos", json={"conteo": 4}).json["stock"] == 4

        assert c.post(f"/api/productos/{pid}/muestras", json={"vector": [0.1] * 1280}).status_code == 201
        assert c.post(f"/api/productos/{pid}/muestras", json={"vector": ["x"] * 100}).status_code == 400
        assert c.post("/api/fondo/muestras", json={"vector": [0.2] * 1280}).status_code == 201
        assert sorted(m["producto_id"] or 0 for m in c.get("/api/muestras").json) == [0, pid]
        assert c.delete("/api/fondo/muestras").status_code == 200
        assert len(c.get("/api/muestras").json) == 1

        prod = c.get("/api/productos").json[0]
        assert (prod["stock"], prod["muestras"]) == (4, 1)

        r = c.get("/api/resumen").json["productos"][0]
        assert r["estado"] == "bajo"                    # 4 <= mínimo 5
        assert r["consumo_diario"] == 0.5               # 7 salidas / 14 días (el ajuste no cuenta)
        assert r["pedir"] == 5                          # ceil(5 + 0.5*7 - 4)

        assert c.delete(f"/api/productos/{pid}").status_code == 200
        assert c.get("/api/muestras").json == []        # se borran en cascada
    print("ok")


if __name__ == "__main__":
    main()
