# StockCam

Clasificador de productos con la cámara del celular. Apuntas a una botella (o a varias), la app reconoce **qué producto es**, cuenta **cuántas hay** y te dice **cuánto queda en inventario**. Con un toque registras la entrada, la salida o el conteo físico.

Materia: Soporte a la toma de decisiones · Autor: _(tu nombre)_

## Cómo funciona

1. **Detección**: COCO-SSD encuentra los objetos en la imagen (botellas, vasos, latas, frutas…).
2. **Clasificación**: cada recorte pasa por MobileNet v2, que lo convierte en un vector de 1280 números. Un clasificador k-NN (k = 5, similitud coseno) lo compara con las muestras que entrenaste y decide qué producto es. Si el parecido queda bajo la **exigencia** (por defecto 75 %), el objeto se marca como no reconocido.
3. **Conteo estable**: se toma la mediana de los últimos 10 cuadros, así una detección suelta no cambia el número.
4. **Inventario**: el stock se calcula como la suma de movimientos (inicial, entrada, salida, ajuste) en SQLite, con historial completo.
5. **Decisión**: el panel calcula el consumo diario (salidas de 14 días), los días de cobertura y cuánto pedir para cubrir 7 días más el stock mínimo.

Todo el reconocimiento corre en el navegador del celular. El servidor solo guarda datos, así que la PC y el celular ven el mismo inventario.

Si el producto tiene código de barras y el navegador lo soporta (Chrome en Android), también lo lee y muestra su stock.

## Ejecutar

```bash
pip install -r requirements.txt
python app.py            # solo en la PC: http://127.0.0.1:5000
python app.py --https    # para usarlo desde el celular
```

Con `--https` la consola muestra la dirección para el celular (misma red Wi-Fi). Como el certificado es autofirmado, el navegador mostrará una advertencia: elige *Avanzado → Continuar*. Sin https el celular no permite usar la cámara.

## Uso

1. **Productos** → crea el producto (nombre, stock inicial, mínimo y, si quieres, código de barras).
2. **Entrenar** → captura 20 a 30 muestras girando el producto. Con dos productos parecidos, entrena ambos.
3. **Escanear** → apunta y usa *+N entrada*, *−N salida* o *Fijar N* (conteo físico).
   Con **Conteo automático** activado, recorres los estantes: cuando el número de un producto se mantiene 2 s, se fija como su inventario (queda como "ajuste" en el historial). Un producto que sale del cuadro no se modifica, y el mismo conteo no se vuelve a registrar.
4. **Panel** → revisa qué está bajo el mínimo y cuánto pedir.

Si confunde productos, sube la exigencia; si no reconoce nada, bájala o agrega muestras.

## Prueba

```bash
python test_app.py
```
