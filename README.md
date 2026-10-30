# Traductor de Lenguaje de Señas a Voz en Tiempo Real

Prototipo modular en Python que traduce el deletreo manual (abecedario de lengua de señas) a texto y voz en tiempo real empleando la cámara web y ejecutándose al 100% en CPU local:

```text
Cámara Web -> Percepción (Manos 21 pts + Cara/Blendshapes + Pose de brazos)
           -> Normalización invariante -> Clasificador (Fase 1 estático / Fase 2 dinámico)
           -> Buffer de estabilidad + Gestos de control -> Post-procesamiento (Gemini opcional)
           -> Síntesis de voz (adaptador a voice-translate)
```

> **Privacidad**: El video nunca se guarda en disco ni se envía a ningún servicio externo. Solo se almacenan vectores numéricos de landmarks en CSV y a Gemini (si se configura clave) únicamente se envía la secuencia de letras en texto plano.

---

## Arquitectura del Proyecto

```text
capture/      -> Gestión de cámara (CAP_DSHOW en Windows) y medición de FPS
perception/   -> MediaPipe Tasks: HandLandmarker, FaceLandmarker (blendshapes) y PoseLandmarker
features/     -> Normalización invariante (traslación, escala y espejado de mano) y ventana temporal
recognition/  -> Recolección a CSV (collect), entrenamiento scikit-learn (train) e inferencia (run)
text/         -> Buffer de estabilidad, bloqueo anti-repetición, gestos de control y Gemini opcional
speech/       -> Adaptador SpeechOutput.speak() reutilizando ./voice-translate sin modificarlo
app/          -> CLI unificada (python -m app [collect|train|run]) y renderizado de overlay OpenCV
tests/        -> Pruebas unitarias deterministas con pytest (sin requerir cámara)
config.yaml   -> Umbrales, alfabetos (ASL y LSM), idioma, modelo Gemini y gestos de control
```

---

## Alfabetos Soportados: Letras Estáticas vs. Dinámicas

Configurables desde `config.yaml` (`alphabet: "ASL"` por defecto; `"LSM"` listo para usar) o por línea de comandos (`--alphabet ASL` / `--alphabet LSM`):

### 1. ASL (*American Sign Language* — por defecto)
- **Estáticas (24 letras — Fase 1)**: `A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y`
- **Dinámicas / con movimiento (2 letras — Fase 2)**: `J` (curva con el meñique), `Z` (trazo en zigzag con el índice)

### 2. LSM (*Lengua de Señas Mexicana*)
- **Estáticas (21 letras — Fase 1)**: `A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y`
- **Dinámicas / con movimiento (6 letras — Fase 2)**: `J, K, Ñ, Q, X, Z`

---

## Requisitos e Instalación

### 1. Entorno Virtual (Python 3.11)
El repositorio reutilizado [`voice-translate`](https://github.com/cccaballero/voice-translate) fija Python 3.11 (`.python-version`) para compatibilidad con Coqui TTS y MediaPipe:

```powershell
# Crear entorno virtual con Python 3.11
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependencias unificadas
pip install -r requirements.txt
```
*(Si usas `uv`: `uv venv --python 3.11 .venv` y `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`).*

### 2. Dependencias del Sistema para `voice-translate`
- **Audio (PortAudio / MCI)**: En Windows, `sounddevice`, `playsound` y el respaldo nativo `winsound` funcionan de serie sin instalar paquetes adicionales. En Linux se requiere `sudo apt install portaudio19-dev espeak-ng` y en macOS `brew install portaudio espeak`.
- **Compatibilidad Windows para Coqui TTS**: El paquete original `TTS==0.22.0` solo publica ruedas precompiladas para Linux y exige *Microsoft Visual C++ 14.0 Build Tools* en Windows. En `requirements.txt` se incluye `coqui-tts` para Windows, que instala el mismo módulo `from TTS.api import TTS` desde ruedas precompiladas (`win_amd64`) sin modificar `voice-translate`.

### 3. Permisos de Cámara en Windows
Si la cámara no abre:
1. Ve a **Configuración de Windows -> Privacidad y seguridad -> Cámara**.
2. Activa **Acceso a la cámara** y **Permitir que las aplicaciones de escritorio accedan a la cámara**.
3. El sistema prueba primero `cv2.CAP_DSHOW` (`use_dshow_on_windows: true` en `config.yaml`) y hace fallback automático a `cv2.CAP_ANY`.

### 4. Post-procesamiento con Gemini (Opcional)
Para corregir secuencias deletreadas (`"H O L A"` -> `"hola"`):
1. Copia `.env.example` a `.env`.
2. Define `GEMINI_API_KEY=tu_clave` y opcionalmente `GEMINI_MODEL=gemini-2.5-flash`.
3. Si no hay clave o no hay conexión, el sistema continúa funcionando localmente sin interrupciones.

---

## Flujo de Uso (`collect` -> `train` -> `run`)

### Paso 1: Capturar tus propias muestras (`collect`)
Como el reconocimiento opera sobre landmarks geométricos normalizados, captura tus muestras frente a la webcam:

```powershell
.\.venv\Scripts\python.exe -m app collect --alphabet ASL --letter A --samples 150
```
- En la ventana interactiva:
  - Pulsa cualquier tecla **`A`–`Z`** o **`Tab`** para elegir la letra objetivo.
  - Pulsa **`Espacio`** para iniciar la captura del lote (150 muestras por defecto para letras estáticas; 20 secuencias de 25 frames para letras dinámicas como `J` y `Z`).
  - Sigue los avisos en pantalla (rotar suavemente el ángulo en el segundo tercio y acercar/alejar la mano en el último tercio).
  - Los datos se guardan incrementalmente en `data/landmarks_static.csv` y `data/landmarks_dynamic.csv`.

### Paso 2: Entrenar el clasificador (`train`)
Una vez capturadas al menos 2 letras distintas:

```powershell
.\.venv\Scripts\python.exe -m app train --classifier mlp
```
- Muestra en consola el **accuracy de validación**, el reporte por clase y la **matriz de confusión**.
- Guarda el modelo estático en `models/sign_classifier.joblib` (y el dinámico en `models/sign_dynamic_classifier.joblib` si capturaste letras dinámicas).

### Paso 3: Traducción en vivo (`run`)
Comando único de arranque:

```powershell
python -m app run
```
*(O con el entorno virtual explícito: `.\.venv\Scripts\python.exe -m app run`).*

#### Controles y Gestos en Vivo (`run`)
| Acción | Gesto Facial / Manual (configurable en `config.yaml`) | Atajo de Teclado |
|---|---|---|
| **Aceptar letra** | Mantener la seña estable durante 12 frames con confianza $\ge 70\%$ | — |
| **Insertar espacio** | Abrir la boca (`mouth_open`) sostenido ~0.8 s | `Espacio` |
| **Borrar última letra** | Fruncir cejas (`brow_down`) sostenido ~0.8 s | `Backspace` |
| **Confirmar y hablar** | Levantar cejas (`brow_up`) o puño izquierdo (`fist_left`) ~0.8 s | `Enter` |
| **Limpiar texto** | — | `C` |
| **Salir** | — | `Q` o `ESC` |

---

## Pruebas Unitarias

Ejecutar la suite de pruebas (normalización invariante de landmarks, trayectorias dinámicas, filtro de estabilidad, bloqueo anti-repetición, gestos y pipeline de entrenamiento):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Limitaciones Conocidas

1. **Confusión entre puños cerrados frontales (`A`, `E`, `M`, `N`, `S`, `T`)**:
   - En una sola vista 2D/2.5D frontal, la diferencia entre `M`, `N`, `T` y `S` depende de bajo qué dedos se asoma la punta del pulgar, lo que sufre auto-oclusión.
   - **Cómo mitigarlo al capturar muestras**: durante `collect`, graba ~50 muestras adicionales de `M`, `N`, `T`, `S`, `A` y `E` inclinando la mano $15^\circ\text{--}25^\circ$ hacia la cámara para que el detector vea claramente la posición del pulgar.
2. **Similitud geométrica entre `U` / `V` / `R` y `G` / `H` / `P` / `Q`**:
   - `U` (índice y medio juntos) y `R` (índice y medio cruzados) tienen distancias entre puntas muy similares; capturar con buena iluminación frontal mejora la separación en el eje X/Z.
3. **Iluminación a contraluz o desenfoque por movimiento rápido**:
   - En señas dinámicas (`J`, `Z`, `Ñ`), un movimiento brusco con poca luz puede hacer perder el tracking de MediaPipe por 1–2 frames.

## Próximos Pasos Sugeridos

1. **Palabras completas y glosas continuas**: Extender la ventana temporal (`TrajectoryBuffer`) para combinar ambas manos + puntos de hombros/cara (`Pose` + `Face`) en un clasificador secuencial (GRU/LSTM ligero o Temporal Convolutional Network) para señas léxicas completas.
2. **Generalización multi-usuario**: Recolectar muestras de 5–10 personas con distintas morfologías de mano y aplicar *data augmentation* geométrico (rotaciones 3D aleatorias de $\pm 15^\circ$ y ruido gaussiano en articulaciones) durante `train`.
3. **Gramática de lengua de señas con Gemini**: Aprovechar los blendshapes faciales (cejas arriba = pregunta sí/no; cejas fruncidas = pregunta *wh-*) como metadatos textuales en el prompt de Gemini para reconstruir la entonación e interrogación al sintetizar la voz.



## Avatar 3D en LSM (`avatar/`)

Avatar de torso de una mujer semirrealista (complexión media femenina, piel morena, ojos azules,
labios llenos, clavículas marcadas, bob corto azul despeinado con raya al lado y mechones sueltos,
top negro de tirantes y pantalón negro) con
brazos y manos rigueados que interpreta texto en español como Lengua de Señas Mexicana.

- **Malla (por defecto `--base makehuman`):** cuerpo y cara realistas a partir de la malla base de
  [MakeHuman](https://github.com/makehumancommunity/makehuman) (CC0, ver
  `avatar/blender/makehuman/LICENSE.ASSETS.md`) con targets de mujer joven mezclados. La columna,
  cuello y cabeza del rig se colocan en las articulaciones de MakeHuman; brazos y manos se reposan de
  la pose A a la pose T del rig (cada hueso se alinea en posición, dirección y longitud). Los pesos
  de calor se calculan en la pose A, con los brazos separados del torso. El top y el pantalón son
  copias suavizadas de la piel, con sus mismos pesos. Unas 123 k caras, GLB de ~6.5 MB.
  `--base procedural` conserva la malla anterior hecha por script (voxel remesh + QuadriFlow).
- **Pesos sin deformaciones:** bone heat + suavizado laplaciano sobre aristas, máx. 4 influencias por
  vértice y normalizados; la blusa copia los pesos de la piel y la piel cubierta se elimina para
  evitar que atraviese la ropa. `tests/test_avatar.py` verifica influencias y huesos del GLB.
- **Movimiento suave:** transiciones con perfil de mínimo jerk y un filtro críticamente amortiguado
  por articulación (`Signer.applySmoothed`) que elimina saltos entre fotogramas clave; cada seña
  mantiene una pausa (`HOLD`) para leerse con claridad.
- **Muñecas naturales:** la pronación/supinación se descompone (swing-twist) y va al hueso de torsión
  del antebrazo; la mano sólo conserva flexión/desviación dentro de `WRIST_LIMITS` (75°/28°, giro
  residual 12°), así no gira de forma imposible.
- **Sin interpenetraciones:** palma y yemas se empujan fuera de volúmenes de cabeza, cuello, pecho y
  abdomen (`Signer.avoidBody`) y las dos manos mantienen una separación mínima.
- **Aspecto:** materiales físicos (piel con *sheen*, cabello metálico con clearcoat, ojos húmedos) e
  iluminación de estudio con `RoomEnvironment`.

```bash
python -m app avatar            # visor en http://127.0.0.1:8765 (también: python -m avatar.server)
```

| Pieza | Archivo |
|---|---|
| Modelo + rig (54 huesos: columna, cuello, cabeza, clavículas, brazos con hueso de torsión, metacarpos y 3 falanges por dedo) generados por script en Blender 4.2 | `avatar/blender/build_avatar.py` → `avatar/web/assets/avatar.glb`, `avatar/data/rig_skeleton.json` |
| Cinemática directa de la mano (21 puntos compatibles con MediaPipe) y rangos articulares naturales/hiperelásticos | `avatar/kinematics.py` |
| "Entrenamiento": ajuste por mínimos cuadrados de ángulos articulares + orientación a landmarks reales | `avatar/train.py` → `avatar/data/handshapes_trained.json` |
| Biblioteca de configuraciones (entrenadas + clasificadores LSM diseñados, incl. hiperextensión) | `avatar/handshapes.py` → `avatar/web/assets/handshapes.json` |
| Texto → glosa LSM (tiempo al inicio, interrogativos al final, lematización) → señas o deletreo | `avatar/lexicon.py`, `avatar/data/lexicon_lsm.json`, `avatar/data/alphabet_lsm.json` |
| Ingesta de diccionarios PDF (PyMuPDF) al formato del léxico | `avatar/ingest_pdf.py` |
| Visor Three.js con IK de dos huesos, orientación de palma/dedos y reparto de pronación entre antebrazo y muñeca | `avatar/web/` |

Regenerar el modelo: `blender -b -P avatar/blender/build_avatar.py -- --out avatar/web/assets/avatar.glb`.

Reentrenar configuraciones: `python -m avatar.train --hf-asl --csv data/landmarks_static.csv --csv-alphabet LSM`
(`--hf-asl` descarga el dataset público *ASL Static Landmarks v1*, CC-BY-4.0; el CSV propio sale de
`python -m app collect`). Después: `python -m avatar.handshapes`.

Ampliar el léxico desde un PDF: `python -m avatar.ingest_pdf diccionario.pdf --fuente "Manos con Voz (SEP/CONAPRED, 2011)"`.
Las entradas sin `keyframes` se animan a partir de `descripcion_movimiento`.

**Limitaciones.** Las configuraciones LSM se inicializan con landmarks de ASL donde la forma de la mano
coincide; conviene recapturar el alfabeto LSM con `python -m app collect` y reentrenar. Las 80 señas del
léxico son descripciones adaptadas (`revisado_por_interprete: false`) y deben validarse con
intérpretes de LSM. No hay expresión facial (componente no manual).
