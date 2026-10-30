# Voz abierta · Frontend

Traductor en vivo en una sola pantalla compartida entre un **alumno sordo que usa Lengua de Señas Mexicana (LSM)** y su **maestro**.

- **Alumno → Maestro:** el alumno seña frente a la cámara y el texto aparece en la conversación del maestro.
- **Maestro → Alumno:** el maestro escribe o dicta. El alumno ve una animación 3D de señas (la genera el backend) y el mensaje en español sencillo.

> Prototipo para el AI Hackathon por México × Clara. La animación de señas es **experimental**: no es una traducción validada por personas sordas ni por intérpretes certificados.

Está hecho con HTML, Sass (BEM) y JavaScript nativo con módulos ES. No usa frameworks. El único paso de compilación es Sass.

---

## 1. Cómo correrlo

Los módulos ES no funcionan con `file://`, así que hay que servir la carpeta por HTTP:

```bash
cd frontend
python -m http.server 8000
# abrir http://localhost:8000
```

La cámara y el micrófono solo funcionan en `localhost` o con HTTPS.

Para el modo real con API, reconocimiento WebSocket y avatar 3D, ejecuta desde la raíz del repositorio:

```bash
python -m app web
# abrir http://127.0.0.1:8000
```

El servidor integrado establece `DEMO_MODE:false`, ofrece `POST /traducir-texto` y recibe cuadros JPEG binarios en `/reconocer`. La traducción incluye `secuencia` (glosa y fotogramas clave) para reproducir señas con el avatar; las respuestas también pueden incluir `animacion_url` para una animación de video.

### Compilar estilos

El `css/main.css` ya viene compilado. Para recompilarlo después de editar el Sass:

```bash
npm install -g sass
sass scss/main.scss css/main.css --style=compressed --source-map
# mientras desarrollas:
sass --watch scss/main.scss css/main.css
```

---

## 2. Estructura

```
frontend/
├── index.html              Traductor (pantalla principal)
├── fuentes.html            Cómo funciona, fuentes candidatas y alcance
├── assets/img/logo.svg
├── css/main.css            Compilado desde scss/ (no editar a mano)
├── scss/
│   ├── main.scss           Índice: orden de importación
│   ├── abstracts/          _tokens (colores, tipografía, espacios, breakpoints), _mixins
│   ├── base/               _reset, _base, _a11y
│   ├── layout/             _header, _footer, _live, _content
│   └── components/         Un archivo por bloque BEM
│       ├── _button  _icon-button  _segmented  _chip  _badge  _avatar  _feedback
│       ├── _panel  _alumno  _stage  _camera  _turn  _subtitle  _confirm
│       ├── _camera-controls  _conversation  _composer  _writer
│       └── _sources  _demo
└── js/
    ├── main.js             Punto de entrada: arma store, controlador y UI
    ├── config.js           Configuración (DEMO_MODE, URLs del backend, tiempos)
    ├── state.js            Estado inicial y store (get / set / subscribe)
    ├── selectors.js        Funciones puras que derivan datos del estado
    ├── controller.js       Toda la lógica: acciones, servicios y temporizadores
    ├── data/demo-data.js   Frases de ejemplo (solo modo demo)
    ├── services/
    │   ├── api.js          Backend: /salud, /traducir-texto, WebSocket de reconocimiento
    │   ├── camera.js       getUserMedia y envío de cuadros
    │   ├── speech.js       Dictado con Web Speech API (es-MX)
    │   └── demo.js         Simulaciones con la misma interfaz que los servicios reales
    ├── ui/
    │   ├── alumno.js       Pinta el panel del alumno
    │   ├── maestro.js      Pinta la conversación y el campo del maestro
    │   ├── feedback.js     Región aria-live, avisos y franja sin conexión
    │   └── demo-panel.js   Panel "Demo" para presentar estados
    └── utils/dom.js        Ayudantes del DOM
```

### Flujo de datos

```
evento (clic, cámara, voz, WebSocket)
   → controller.js  actualiza el estado con store.set()
   → store notifica → ui/*.js vuelve a pintar a partir del estado (selectors.js)
```

- Los botones declaran su acción con `data-action="…"`. Un único manejador en `main.js` llama a `controller.actions[…]`.
- Los elementos que JavaScript necesita encontrar usan atributos `data-*`, no clases. **Las clases son solo para estilos.**

---

## 3. Convenciones de CSS (BEM)

| Tipo | Ejemplo |
|---|---|
| Bloque | `.composer` |
| Elemento | `.composer__input`, `.composer__send` |
| Modificador | `.composer--listening`, `.button--primary` |

- Cada bloque está en su propio archivo dentro de `scss/components/`.
- Los estados de interacción usan **atributos ARIA**, no clases `is-*`: `[aria-pressed="true"]`, `[aria-current="page"]`, `[hidden]`, `:disabled`.
- Los colores de cada zona se manejan con propiedades personalizadas (`--fg`, `--fg-2`, `--ring`, `--line-strong`…) a través de los mixins `context-light`, `context-surface` y `context-dark`.
- Los paneles usan *container queries* (`container-name: panel`). La versión de escritorio sin scroll se activa con `min-width: 1000px` y `min-height: 600px`.

### Paleta y contraste (WCAG)

| Combinación | Contraste | Uso |
|---|---|---|
| Obsidiana `#0B0D12` sobre Blanco `#F2F4F7` | 17.6:1 | Texto de la página |
| Blanco sobre Azul metálico `#344B68` | 8.1:1 | Texto en paneles |
| Neón `#B6FF3B` sobre Azul | 7.4:1 | Foco y selección en paneles |
| Obsidiana sobre Neón | 16.1:1 | Botón principal |
| `#4A5263` sobre Blanco | 7.1:1 | Texto secundario |
| `#C3CBD7` sobre Azul | 5.5:1 | Texto secundario en paneles |
| Neón sobre Blanco | 1.1:1 | **Nunca usar** |

---

## 4. Configuración

Toda la configuración está en `js/config.js`. Puedes sobrescribirla sin tocar el código, antes de cargar `main.js`:

```html
<script>
  window.AULA_CONFIG = {
    DEMO_MODE: false,
    API_URL: 'https://api.ejemplo.mx',
    RECOGNITION_WS_URL: 'wss://api.ejemplo.mx/reconocer',
  };
</script>
```

| Clave | Valor por defecto | Para qué sirve |
|---|---|---|
| `DEMO_MODE` | `true` | Simula cámara, reconocimiento y animación, y muestra el botón **Demo** |
| `API_URL` | `''` | Base del backend (vacío = mismo origen) |
| `RECOGNITION_WS_URL` | `''` | WebSocket del reconocimiento de LSM |
| `MAX_MESSAGE_LENGTH` | `300` | Máximo de caracteres por mensaje del maestro |
| `AUTO_SEND_DELAY_MS` | `1400` | Tiempo para cancelar antes de enviar señas o dictado |
| `FRAME_INTERVAL_MS` | `100` | Intervalo de envío de cuadros JPEG al reconocimiento |

**No pongas llaves de API en el frontend.** Las llamadas a modelos de IA se hacen desde el backend.

---

## 5. Contrato con el backend (Python)

### `GET /salud`

Comprueba la conexión. Responde `200` si el servicio está disponible.

### `POST /traducir-texto`

Convierte el texto del maestro a español sencillo y a una animación de señas.

```json
// Petición
{ "texto": "Es el proceso con el que las plantas producen azúcares usando la luz." }

// Respuesta
{
  "texto_sencillo": "Las plantas usan la luz para hacer azúcar.",
  "glosa": "PLANTA LUZ AZUCAR",
  "secuencia": [{"tipo": "sena", "glosa": "PLANTA", "keyframes": []}],
  "animacion_url": null
}
```

La respuesta contiene una `secuencia` de señas para el avatar. `animacion_url` puede ser `null`; cuando se envía una URL de video, el frontend conserva el reproductor de video.

### WebSocket de reconocimiento (`RECOGNITION_WS_URL`)

El frontend envía cuadros de la cámara como JPEG (binario) por `/reconocer`. El backend responde con mensajes JSON:

| `tipo` | Campos | Qué hace el frontend |
|---|---|---|
| `manos` | `detectadas: bool` | Marca "Manos detectadas" |
| `parcial` | `texto` | Muestra el texto mientras el alumno seña |
| `final` | `texto` | Muestra "Entendimos: …" y lo envía después de 1.4 s si nadie lo cancela |
| `no_reconocido` | — | Pide repetir o escribir |
| `calidad` | `encuadre: bool`, `luz: bool` | Actualiza la lista de comprobación de la cámara |

### Servir el frontend desde Python

**FastAPI**

```python
from fastapi.staticfiles import StaticFiles
# Registrar las rutas de la API antes de montar los estáticos
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

**Flask**

```python
app = Flask(__name__, static_folder="frontend", static_url_path="")

@app.get("/")
def index():
    return app.send_static_file("index.html")
```

Si el frontend y el backend están en dominios distintos, el backend debe permitir CORS para el origen del frontend.

---

## 6. Accesibilidad

- Todo funciona con teclado. El foco es visible: neón en paneles y obsidiana sobre la página.
- La región `aria-live` anuncia los mensajes nuevos, los cambios de turno y los errores.
- El estado nunca depende solo del color: siempre hay texto o ícono.
- Respeta `prefers-reduced-motion`.
- El alumno puede ver el texto original, agrandar el texto (A+), cambiar la velocidad de la animación o escribir en lugar de señar.
- La cámara se puede pausar, ocultar o mostrar solo como puntos.

## 7. Privacidad

- El video de la cámara no se guarda. En modo demo, ni siquiera se usa la cámara real.
- La conversación vive solo en la memoria del navegador y se borra al recargar.

## 8. Honestidad del prototipo

- Las respuestas del modo demo son de ejemplo y lo dicen en pantalla.
- La página **Fuentes** lista fuentes *candidatas*. No están integradas ni validadas todavía.
- Ninguna salida de LSM se presenta como traducción validada.
