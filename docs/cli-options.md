# Guía detallada de opciones de `zoom.py`

Este documento reúne la descripción de cada bandera disponible en `zoom.py`, explica para qué sirve y enlaza un comando de ejemplo reproducible. Todos los ejemplos se ejecutaron con `scripts/generate_cli_examples.py`, que también deposita las capturas resultantes bajo `examples/cli-options/`. Si necesitas volver a generarlas, ejecuta:

```bash
python scripts/generate_cli_examples.py
```

Cada sección incluye un comando listo para copiar, acompañado del artefacto que deberías obtener dentro del repositorio.

## Geometría y animación

### `--max-iterations`
Controla cuántas veces se itera el mapa de Mandelbrot por píxel. Valores altos resuelven filamentos finos, pero aumentan el tiempo de renderizado.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --max-iterations 3500 \
  --output examples/cli-options/max-iterations/high-iterations.png
```
Resultado: `examples/cli-options/max-iterations/high-iterations.png`

### `--x-res`
Define el número de columnas de la imagen renderizada. Útil para ajustar la nitidez o acelerar pruebas reduciendo la resolución horizontal.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 240 --y-res 160 \
  --output examples/cli-options/x-res/wide-resolution.png
```
Resultado: `examples/cli-options/x-res/wide-resolution.png`

### `--y-res`
Define el número de filas de la imagen. Puedes comprimir o expandir la altura para adaptarte a diferentes formatos de vídeo.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 96 \
  --output examples/cli-options/y-res/short-resolution.png
```
Resultado: `examples/cli-options/y-res/short-resolution.png`

### `--x-center`
Desplaza el centro del encuadre sobre el eje real del plano complejo para explorar estructuras específicas.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --x-center -1.401155 \
  --output examples/cli-options/x-center/period-three.png
```
Resultado: `examples/cli-options/x-center/period-three.png`

### `--x-width`
Indica el ancho inicial de la ventana en el plano complejo. Valores pequeños arrancan más cerca del conjunto.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --x-width 0.6 \
  --output examples/cli-options/x-width/narrow-window.png
```
Resultado: `examples/cli-options/x-width/narrow-window.png`

### `--y-center`
Ajusta el centro del encuadre sobre el eje imaginario para desplazarte verticalmente por el conjunto.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --y-center 0.35 \
  --output examples/cli-options/y-center/upper-plane.png
```
Resultado: `examples/cli-options/y-center/upper-plane.png`

### `--y-width`
Especifica la altura inicial del área muestreada en el plano complejo.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --y-width 1.2 \
  --output examples/cli-options/y-width/tall-window.png
```
Resultado: `examples/cli-options/y-width/tall-window.png`

### `--lock-aspect`
Mantiene `y_width = x_width × (y_res / x_res)` en cada cuadro para evitar que las circunferencias se deformen cuando solo modificas la resolución horizontal.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 256 --y-res 160 \
  --lock-aspect \
  --output examples/cli-options/lock-aspect/locked.png
```
Resultado: `examples/cli-options/lock-aspect/locked.png`

### `--zoom-factor`
Multiplica el tamaño de la ventana en cada cuadro. Valores menores que 1 acercan el zoom; mayores que 1 alejan la cámara.

**Ejemplo**
```bash
python zoom.py --frames 6 --mode image --x-res 160 --y-res 160 \
  --zoom-factor 0.7 \
  --output examples/cli-options/zoom-factor/fast-zoom.png
```
Resultado: `examples/cli-options/zoom-factor/fast-zoom.png`

### `--final-zoom`
Define la escala total del último cuadro. El planificador calcula el factor por cuadro y sustituye a `--zoom-factor`.

**Ejemplo**
```bash
python zoom.py --frames 5 --mode image --x-res 160 --y-res 160 \
  --final-zoom 1e-3 \
  --output examples/cli-options/final-zoom/target-scale.png
```
Resultado: `examples/cli-options/final-zoom/target-scale.png`

### `--easing`
Selecciona la curva temporal del zoom (`ease`, `linear`, etc.). Permite controlar la aceleración y desaceleración.

**Ejemplo**
```bash
python zoom.py --frames 6 --mode image --x-res 160 --y-res 160 \
  --zoom-factor 0.7 --easing linear \
  --output examples/cli-options/easing/linear-ease.png
```
Resultado: `examples/cli-options/easing/linear-ease.png`

### `--frames`
Cantidad de cuadros que se renderizan. Afecta tanto a la duración de GIFs como al número de imágenes intermedias.

**Ejemplo**
```bash
python zoom.py --frames 12 --mode gif --x-res 160 --y-res 160 \
  --zoom-factor 0.85 \
  --output examples/cli-options/frames/twelve-frames.gif
```
Resultado: `examples/cli-options/frames/twelve-frames.gif`

## Gestión de salida

### `--mode`
Permite seleccionar múltiples modos de salida (`gif`, `image`, `frames`, `mono`). Repite la bandera para añadir más de uno.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --mode mono \
  --frame-dir examples/cli-options/mode/mono \
  --x-res 160 --y-res 160 \
  --output examples/cli-options/mode/single-frame.png
```
Resultados: `examples/cli-options/mode/single-frame.png` y la secuencia en `examples/cli-options/mode/mono/`.

### `--output`
Sobrescribe la ruta del archivo de salida cuando generas un único GIF o imagen.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --output examples/cli-options/output/custom-name.png
```
Resultado: `examples/cli-options/output/custom-name.png`

### `--frame-dir`
Define el directorio donde se guardan los fotogramas individuales cuando seleccionas `--mode frames` o `--mode mono`.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode frames \
  --frame-dir examples/cli-options/frame-dir/frames \
  --x-res 160 --y-res 160
```
Resultado: secuencia en `examples/cli-options/frame-dir/frames/`.

### `--keep-frames`
Mantiene los fotogramas temporales que se utilizan para construir un GIF.

**Ejemplo**
```bash
python zoom.py --frames 4 --mode gif --keep-frames \
  --frame-dir examples/cli-options/keep-frames/frames \
  --x-res 160 --y-res 160 \
  --output examples/cli-options/keep-frames/zoom.gif
```
Resultados: `examples/cli-options/keep-frames/zoom.gif` y los cuadros intermedios en `examples/cli-options/keep-frames/frames/`.

### `--gif-frame-duration`
Controla cuánto tiempo (en segundos) permanece cada cuadro en pantalla dentro del GIF.

**Ejemplo**
```bash
python zoom.py --frames 6 --mode gif --x-res 160 --y-res 160 \
  --gif-frame-duration 0.2 \
  --output examples/cli-options/gif-frame-duration/slow.gif
```
Resultado: `examples/cli-options/gif-frame-duration/slow.gif`

### `--save-frames` *(obsoleta)*
Activa el modo histórico equivalente a `--mode frames`. No puede combinarse con `--mode`.

**Ejemplo**
```bash
python zoom.py --frames 3 --save-frames \
  --frame-dir examples/cli-options/save-frames/frames \
  --x-res 160 --y-res 160
```
Resultado: secuencia en `examples/cli-options/save-frames/frames/` (se emite una advertencia de deprecación).

### `--save-mono` *(obsoleta)*
Antiguo alias de `--mode mono`. Mantiene la salida en escala de grises.

**Ejemplo**
```bash
python zoom.py --frames 3 --save-mono \
  --frame-dir examples/cli-options/save-mono/mono \
  --x-res 160 --y-res 160
```
Resultado: secuencia monocromática en `examples/cli-options/save-mono/mono/` (con advertencia de deprecación).

### `--colormap`
Selecciona el mapa de color de Matplotlib usado para colorear los valores suavizados.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --colormap inferno \
  --output examples/cli-options/colormap/inferno.png
```
Resultado: `examples/cli-options/colormap/inferno.png`

### `--format`
Especifica la extensión del archivo cuando se genera una imagen fija (cualquier formato soportado por Pillow).

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --format webp \
  --output examples/cli-options/format/custom.webp
```
Resultado: `examples/cli-options/format/custom.webp`

### `--frames-path` *(alias)*
Alias antiguo de `--frame-dir`. Útil para reproducir scripts existentes.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode frames \
  --frames-path examples/cli-options/frames-path/frames \
  --x-res 160 --y-res 160
```
Resultado: `examples/cli-options/frames-path/frames/`.

## Visualización y anotaciones

### `--show-edges`
Muestra el resultado del detector de bordes (Sobel) junto a la imagen coloreada.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --show-edges \
  --output examples/cli-options/show-edges/edges.png
```
Resultado: `examples/cli-options/show-edges/edges.png`

### `--show-coordinates`
Superpone los límites del plano complejo y el origen en la imagen final.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --show-coordinates \
  --output examples/cli-options/show-coordinates/annotated.png
```
Resultado: `examples/cli-options/show-coordinates/annotated.png`

## Mapeo de tonos

### `--normalize {outside,all}`
Determina qué muestras aportan al histograma de normalización. `outside` usa solo los puntos que escapan; `all` considera todo.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --normalize all \
  --output examples/cli-options/normalize/all-samples.png
```
Resultado: `examples/cli-options/normalize/all-samples.png`

### `--gamma`
Ajusta la corrección gamma tras la normalización. Valores mayores que 1 aclaran medios tonos.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --gamma 1.1 \
  --output examples/cli-options/gamma/bright.png
```
Resultado: `examples/cli-options/gamma/bright.png`

### `--clip-low`
Marca el percentil inferior que se recorta antes de normalizar, evitando que los valores más oscuros dominen la escala.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --clip-low 5 \
  --output examples/cli-options/clip-low/higher-floor.png
```
Resultado: `examples/cli-options/clip-low/higher-floor.png`

### `--clip-high`
Marca el percentil superior que se recorta para proteger las altas luces.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --clip-high 90 \
  --output examples/cli-options/clip-high/lower-ceiling.png
```
Resultado: `examples/cli-options/clip-high/lower-ceiling.png`

### `--tone-smoothing`
Suaviza en el tiempo los percentiles de corte para reducir parpadeos entre cuadros sucesivos.

**Ejemplo**
```bash
python zoom.py --frames 5 --mode image --x-res 160 --y-res 160 \
  --tone-smoothing 0.6 \
  --output examples/cli-options/tone-smoothing/smoothed.png
```
Resultado: `examples/cli-options/tone-smoothing/smoothed.png`

### `--invert`
Invierte el mapa de color seleccionado.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --invert \
  --output examples/cli-options/invert/inverted.png
```
Resultado: `examples/cli-options/invert/inverted.png`

### `--inside-color`
Define el color (hexadecimal) usado para los puntos que permanecen dentro del conjunto de Mandelbrot.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --inside-color #0a3ba0 \
  --output examples/cli-options/inside-color/custom-interior.png
```
Resultado: `examples/cli-options/inside-color/custom-interior.png`

## Diagnóstico

### `-v`, `--verbose`
Activa registros adicionales de TensorFlow y detecta hardware disponible, útil para depurar configuraciones.

**Ejemplo**
```bash
python zoom.py --frames 1 --mode image --x-res 160 --y-res 160 \
  --verbose \
  --output examples/cli-options/verbose/diagnostic.png
```
Resultado: `examples/cli-options/verbose/diagnostic.png` (la consola muestra información adicional sobre la CPU/GPU).
