# Taller Final IA: OCR + LLM

**Curso:** Inteligencia Artificial
**Universidad:** EAFIT
**Profesor:** Jorge Padilla
**Integrantes del Grupo**

- Juan Pablo Duque
- Sebastián Villegas
- Gabriela Martínez

---

## Aplicación Pública Desplegada

Este proyecto es una aplicación web multimodal que integra la Visión Artificial para la extracción de texto (OCR) y el Procesamiento de Lenguaje Natural (NLP) para su análisis. La aplicación está disponible públicamente en Streamlit Community Cloud.

**[Enlace Público de Entrega](https://finalproject-ai-ahvnxq3xadrljrcg7pscsn.streamlit.app/)**

---

## Cumplimiento de Objetivos

El objetivo de esta aplicación es:
* Permitir la carga de una imagen.
* Extraer automáticamente el texto usando un modelo OCR.
* Enviar el texto a un Modelo de Lenguaje Grande (LLM) para realizar tareas como resumir, traducir o analizar.
* Permitir cambiar de proveedor de LLM (GROQ vs. Hugging Face) y ajustar parámetros clave.

### Tecnologías Clave

* **Interfaz:** `streamlit`
* **OCR:** `easyocr` (cargado una sola vez con `@st.cache_resource` para optimizar el rendimiento)
* **LLM API:** `groq` (Para velocidad y accesibilidad a modelos potentes)
* **Modelos NLP:** `transformers` (Usados localmente para las tareas de Hugging Face)

---

## Notas de Implementación (Desviaciones y Faltantes)

Se tomaron las siguientes consideraciones que difieren ligeramente de las especificaciones originales del taller:

### 1. Desviación: Implementación de Hugging Face
El taller sugería usar la **API de Inferencia** de Hugging Face (`InferenceClient`).
* **Lo implementado:** Se optó por ejecutar los modelos de `transformers` **localmente** para las tareas de resumen, entidad y traducción. Esto demuestra la capacidad de integrar modelos *serverless* locales y no requiere una clave API de Hugging Face.

### 2. Faltante: Control de `max_tokens`
El taller solicitaba un control interactivo de los parámetros `temperature` y `max_tokens`.
* **Lo implementado:** Solo se incluyó el control de `temperature`. En su lugar, se añadió un *slider* para limitar el "Máx. caracteres a analizar" (controlando la longitud de la **entrada**), en lugar del `max_tokens` (que controla la longitud de la **salida**).