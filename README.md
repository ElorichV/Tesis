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

