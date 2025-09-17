# Estrategia de salida: estado actual y rediseño propuesto

## 1. Flujo de salida existente

### 1.1 Generación del GIF animado
- Cada cuadro renderizado se almacena en memoria en la lista `images` dentro del bucle principal de renderizado.
- Al terminar el bucle, el primer elemento de `images` se usa para escribir `movie.gif` en el directorio de trabajo. El resto de la lista se pasa como `append_images`, con una duración fija de 100 ms por cuadro y `loop=0` para repetir indefinidamente.
- El nombre del archivo (`movie.gif`) y la duración del cuadro están codificados; no existe un flag CLI para cambiarlos.

### 1.2 Guardado opcional de cuadros individuales
- Los flags `--save-frames` y `--save-mono` activan la escritura en disco de los cuadros coloreados y/o monocromáticos.
- Cuando alguno de estos flags está activo se crea (si es necesario) el directorio indicado por `--frames-path` (por defecto `./frames`).
- Las imágenes coloreadas se guardan como `frame{NNN}.{format}`; el formato se controla con `--format` (valor por defecto `png`).
- Las imágenes monocromáticas se guardan como `mono{NNN}.{format}` reutilizando el mismo formato indicado por `--format`.
- Si `--show-edges` está activo, cada archivo `frame{NNN}` contiene la composición de la imagen RGB y la visualización de bordes concatenada.

### 1.3 Otros parámetros relevantes
- `--save-frames` y `--save-mono` no afectan a la generación del GIF: este se produce siempre.
- No hay forma de evitar la acumulación en memoria de todos los cuadros, ya que la lista `images` se necesita para guardar el GIF al final.
- Tampoco existen rutas configurables para el GIF ni para un único fotograma final.

## 2. Limitaciones detectadas
1. **Salida fija**: el flujo actual siempre produce `movie.gif`, incluso cuando el usuario solo desea un fotograma o una secuencia de archivos.
2. **Flags superpuestos**: `--save-frames`, `--save-mono` y `--frames-path` representan modos de salida, pero se expresan como flags independientes que pueden entrar en combinaciones incoherentes.
3. **Acumulación de memoria**: la necesidad de guardar todos los cuadros para crear el GIF puede ser un problema con secuencias largas.

## 3. Matriz de modos propuesta
Se introduce un flag `--mode` (repetible) que define explícitamente qué artefactos producir. Valores válidos:

| Modo | Propósito | Artefacto generado | Salida por defecto |
|------|-----------|--------------------|--------------------|
| `gif` | Animación completa | Archivo `.gif` con todos los cuadros renderizados | `movie.gif` en el directorio de trabajo |
| `image` | Fotograma final coloreado | Archivo de imagen único usando el formato activo (`--format`) | `frame_final.<format>` en el directorio de trabajo |
| `frames` | Secuencia de cuadros coloreados | Directorio con archivos `frame{NNN}.<format>` | Subdirectorio `frames/` |
| `mono` | Secuencia de cuadros monocromáticos | Directorio con archivos `mono{NNN}.<format>` | Subdirectorio `frames/` (compartido con `frames`) |

Notas:
- `--mode` puede especificarse varias veces (`--mode gif --mode frames`). Si no se indica, el valor por defecto es `gif` para mantener compatibilidad.
- El modo `mono` reemplaza al flag histórico `--save-mono`.

## 4. Flags nuevos o modificados

| Flag | Tipo | Descripción | Por defecto | Modos requeridos |
|------|------|-------------|-------------|------------------|
| `--mode MODE` | Repetible | Registra un modo de salida. Valores válidos: `gif`, `image`, `frames`, `mono`. | `[gif]` | — |
| `--output PATH` | Ruta | Destino del artefacto principal. Si solo hay un modo con archivo (`gif` o `image`), debe ser una ruta de archivo. Con múltiples modos basados en archivo, debe ser un directorio contenedor. | Depende del modo (ver §3) | `gif`, `image` |
| `--frame-dir DIR` | Ruta | Directorio donde se escriben los cuadros cuando los modos `frames` o `mono` están activos. Se crea si no existe. | `frames` | `frames`, `mono`, `gif` (solo si se activa `--keep-frames`) |
| `--keep-frames` | Flag | Para el modo `gif`, solicita conservar los cuadros individuales en `--frame-dir` además del GIF final. | Desactivado | `gif` |
| `--format EXT` | Cadena | Extensión para los archivos de imagen (`png`, `jpeg`, etc.). Se aplica a `image`, `frames` y `mono`. | `png` | `image`, `frames`, `mono` |

## 5. Reglas y validaciones

1. **Selección de modos**
   - Error si se especifica un modo no reconocido.
   - Error si `--mode` queda vacío (p. ej. por lectura desde archivo de configuración). El asistente debe imponer el valor por defecto `gif`.
2. **Uso de `--output`**
   - Cuando solo se solicita un modo basado en archivo (`gif` o `image`), `--output` debe ser una ruta a archivo válida; si apunta a un directorio se considera error.
   - Cuando se solicitan ambos (`gif` e `image`), `--output` debe ser un directorio que contenga ambos artefactos (`movie.gif` y `frame_final.<format>`). Si no se proporciona, se usa el directorio de trabajo actual.
   - `--output` no es válido si el único modo solicitado es `frames` o `mono`.
3. **Uso de `--frame-dir`**
   - Requerido (con valor por defecto) cuando `frames` o `mono` estén activos.
   - Provoca error si se especifica sin incluir alguno de esos modos, salvo que se combine con `--keep-frames` y `gif`.
   - El proceso debe crear el directorio si no existe, o validar que se puede escribir en él.
4. **Extensiones y formatos**
   - Para `image`, si `--output` incluye una extensión, debe coincidir con `--format` (o se emite un error/configuración automática documentada).
   - Para `gif`, el archivo de salida debe terminar en `.gif`.
   - `--format` no afecta al modo `gif`.
5. **Compatibilidad con flags anteriores**
   - `--save-frames` pasa a ser alias de `--mode frames` (con aviso de deprecación).
   - `--save-mono` pasa a ser alias de `--mode mono`.
   - `--frames-path` se mantiene como alias de `--frame-dir`.
   - Se debe rechazar la combinación de los alias antiguos con los flags nuevos cuando generen ambigüedad (por ejemplo, `--save-frames --mode image --frame-dir otra_ruta`).
6. **Acumulación de memoria**
   - Para `gif`, se recomienda mantener una política que permita escribir cuadros temporales a disco si `--keep-frames` o `frames` están activos, evitando retener toda la animación en memoria.

## 6. Casos de uso ilustrativos

| Escenario | Comando esperado | Resultados |
|-----------|------------------|------------|
| Animación estándar | `python zoom.py` | Genera `movie.gif` en el directorio de trabajo |
| Solo fotograma final PNG | `python zoom.py --mode image --format png --output renders/final.png` | Crea `renders/final.png`; no se producen GIF ni secuencias |
| GIF + secuencia de cuadros | `python zoom.py --mode gif --mode frames --frame-dir ./frames` | Guarda `movie.gif` y `./frames/frameNNN.png` |
| Secuencia monocromática | `python zoom.py --mode mono --frame-dir ./mono_frames --format png` | Genera `./mono_frames/monoNNN.png` |
| GIF sin conservar cuadros | `python zoom.py --mode gif` | Produce únicamente `movie.gif`; no se crea `frames/` |

Estas reglas servirán como referencia para las próximas tareas de implementación del nuevo sistema de salida.
