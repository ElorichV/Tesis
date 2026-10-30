/**
 * Datos de demostración (ficticios). Se usan solo con CONFIG.DEMO_MODE.
 */

/** Conversación de ejemplo en una clase de ciencias. */
export const EXAMPLE_CONVERSATION = [
  { author: 'maestro', via: 'texto', text: 'Hoy vamos a ver la fotosíntesis.', time: '10:02', animation: 'unavailable' },
  { author: 'alumno', via: 'lsm', text: '¿Qué es la fotosíntesis?', time: '10:02' },
  { author: 'maestro', via: 'texto', text: 'Es el proceso con el que las plantas producen azúcares usando la luz.', time: '10:03', animation: 'unavailable' },
  { author: 'alumno', via: 'lsm', text: '¿Puede repetir más despacio?', time: '10:03' },
];

/** Frases que «reconoce» la cámara simulada, en orden. */
export const SIGN_SCRIPT = [
  '¿Qué es la fotosíntesis?',
  '¿Puede repetir más despacio?',
  'Ya entendí, gracias.',
];

/** Frases que «dicta» el maestro simulado, en orden. */
export const DICTATION_SCRIPT = [
  'Muy bien. Ahora pongan atención.',
  'Las plantas necesitan luz, agua y aire.',
];

/**
 * Versión en español sencillo que entregaría el servicio para el alumno.
 * En producción viene en la respuesta del backend (campo texto_sencillo).
 */
export const SIMPLE_SPANISH = {
  'Hoy vamos a ver la fotosíntesis.': 'Hoy vamos a estudiar la fotosíntesis.',
  'Es el proceso con el que las plantas producen azúcares usando la luz.': 'Las plantas usan la luz para hacer azúcar.',
  'Muy bien. Ahora pongan atención.': 'Muy bien. Ahora pon atención.',
  'Las plantas necesitan luz, agua y aire.': 'Las plantas necesitan luz, agua y aire.',
};
