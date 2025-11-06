import os
import io
import re
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ============================
# Configuración básica (UI)
# ============================
st.set_page_config(page_title="Taller IA: OCR + LLM", layout="centered")

# Estilos globales 
st.markdown("""
<style>
/* Ocultar menú y footer de Streamlit para un look más limpio */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Tipografía y contenedores */
html, body, [class*="css"]  {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.section {
  padding: 1.25rem 1.25rem;
  border: 1px solid #e7e7e9;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
h1.page-title {
  font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem;
}
.subtle {
  color: #6B7280; font-size: 0.95rem; margin-bottom: 1.25rem;
}
.block-label {
  font-weight: 600; margin-bottom: 0.35rem;
}
hr.divider {
  border: none; border-top: 1px solid #e7e7e9; margin: 1.25rem 0;
}
.result-box {
  padding: 0.85rem 1rem; background: #F9FAFB; border: 1px solid #ECEFF3; border-radius: 10px;
  white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Taller IA: OCR + LLM</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Taller Final (OCR + análisis con modelos de lenguaje)</div>', unsafe_allow_html=True)

# Carga variables .env si estás en local
load_dotenv(override=True)

# ============================
# Parámetros internos (no visibles)
# ============================
DEFAULT_GROQ_MAX_TOKENS = 1024   # límite razonable para no romper el cliente
DEFAULT_HF_MAX_TOKENS   = 160    # rápido y suficiente para las tareas

# ============================
# Helpers: OCR (EasyOCR)
# ============================
@st.cache_resource(show_spinner=False)
def load_ocr_reader(langs=("es", "en")):
    """Carga el modelo de EasyOCR una sola vez (cacheado)."""
    try:
        import easyocr  # noqa
        return easyocr.Reader(list(langs))
    except Exception as e:
        st.error(f"No se pudo cargar EasyOCR. Verifica dependencias. Error: {e}")
        return None

def run_easyocr(reader, image_bytes):
    if reader is None:
        return ""
    import numpy as np  # noqa
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # easyocr espera un array numpy BGR o ruta de archivo
    np_img = np.array(image)[:, :, ::-1]  # RGB -> BGR
    result = reader.readtext(np_img, detail=0, paragraph=True)
    return "\n".join(result).strip()

# ============================
# Helpers: GROQ (chat.completions)
# ============================
def groq_chat_completion(prompt, model_name, temperature=0.3, max_tokens=DEFAULT_GROQ_MAX_TOKENS):
    """Llama a la API de GROQ con manejo de deprecaciones y fallback automático."""
    try:
        from groq import Groq  # noqa
    except Exception:
        st.error("No se encontró el SDK de GROQ. Agrega 'groq' a requirements.txt")
        raise

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Falta GROQ_API_KEY en tus variables de entorno / secretos de Streamlit.")
        return ""

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": "Eres un asistente útil. Responde de forma clara y concisa en el mismo idioma del usuario."},
        {"role": "user", "content": prompt},
    ]

    def _call(model):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    try:
        resp = _call(model_name)
        return resp.choices[0].message.content
    except Exception as e:
        # Si el modelo fue retirado, hacemos fallback
        if "decommissioned" in str(e) or "model_decommissioned" in str(e):
            fallback = "llama-3.1-8b-instant"
            try:
                resp = _call(fallback)
                st.warning(f"Modelo no disponible. Usé {fallback} automáticamente.")
                return resp.choices[0].message.content
            except Exception as e2:
                st.error(f"Fallo en GROQ (fallback): {e2}")
                return ""
        else:
            st.error(f"Fallo en GROQ: {e}")
            return ""

# ============================
# Helpers: Hugging Face (Transformers local rápidos)
# ============================
def _clean_ticks(s: str) -> str:
    """Limpia horas/timestamps tipo 10:30 y normaliza bullets/espacios."""
    s = re.sub(r"\b\d{1,2}:\d{2}\b", "", s)
    s = re.sub(r"[•\-\u2022]+\s*", "• ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

@st.cache_resource(show_spinner=False)
def _local_summarizer_fast():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-base")  # más rápido que bart-large

def hf_summarize(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    """Resumen rápido: BART-base local."""
    try:
        pipe = _local_summarizer_fast()
        out = pipe(text, max_length=min(256, int(max_tokens)), do_sample=False)
        if isinstance(out, list) and out and "summary_text" in out[0]:
            return out[0]["summary_text"]
        return str(out)
    except Exception as e:
        st.error(f"Resumen rápido falló: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def _local_ner_fast():
    from transformers import pipeline
    return pipeline(
        "token-classification",
        model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
        aggregation_strategy="simple",
    )

def hf_entities(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    """Extracción rápida de entidades (local)."""
    try:
        ner = _local_ner_fast()
        ents = ner(text)
        cat_map = {
            "PER": "PERSONA",
            "ORG": "ORGANIZACIÓN",
            "LOC": "LUGAR",
            "MISC": "OTRA",
            "DATE": "FECHA",
        }
        lines = []
        for e in ents:
            label = e.get("entity_group", "MISC")
            cat = cat_map.get(label, label)
            val = e.get("word", "").strip()
            if val:
                lines.append(f"• [{cat}]: {val}")
        return "\n".join(lines) if lines else "• [INFO]: No se detectaron entidades claras."
    except Exception as e:
        st.error(f"NER rápido falló: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def _local_translator_es_en_fast():
    from transformers import pipeline
    return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def hf_translate_to_english(text, max_tokens=DEFAULT_HF_MAX_TOKENS):
    """Traducción rápida (local)."""
    try:
        translator = _local_translator_es_en_fast()
        out = translator(text, max_length=min(256, int(max_tokens)))
        if isinstance(out, list) and out and "translation_text" in out[0]:
            return out[0]["translation_text"]
        return str(out)
    except Exception as e:
        st.error(f"Traducción local falló: {e}")
        return ""

# ============================
# UI
# ============================

# Sección: OCR
with st.container():
    st.markdown("### Módulo 1 · OCR")
    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Sube una imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])
    col1, col2 = st.columns([1, 1])

    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""

    with col1:
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            # Soporte para distintas versiones de Streamlit
            try:
                st.image(img, caption="Imagen subida", use_container_width=True)
            except TypeError:
                st.image(img, caption="Imagen subida", use_column_width=True)

    with col2:
        if st.button("Extraer texto (EasyOCR)", use_container_width=True):
            with st.spinner("Leyendo texto con EasyOCR..."):
                reader = load_ocr_reader()
                st.session_state.extracted_text = run_easyocr(reader, uploaded.getvalue())

    st.text_area("Texto extraído", key="extracted_text", height=220, label_visibility="visible", placeholder="Aquí verás el texto del OCR...")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider" />', unsafe_allow_html=True)

# Sección: LLM
with st.container():
    st.markdown("### Módulos 2 y 3 · Análisis con LLM")
    st.markdown('<div class="section">', unsafe_allow_html=True)

    # Controles de proveedor y parámetros
    colp, colc = st.columns([1, 1])
    with colp:
        provider = st.radio("Proveedor de LLM", ["GROQ", "Hugging Face"], horizontal=True)
    with colc:
        max_chars = st.slider("Máx. caracteres a analizar", 500, 6000, 2500, 100)

    temperature = st.slider("Creatividad (GROQ · temperature)", 0.0, 1.5, 0.3, 0.05)
    task = st.selectbox(
        "Tarea a realizar sobre el texto",
        ["Resumir en 3 puntos clave", "Identificar entidades principales", "Traducir al inglés"],
    )

    if provider == "GROQ":
        groq_model = st.selectbox(
            "Modelo de GROQ",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            index=0,
        )

    run = st.button("Analizar texto", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Resultado
    if run:
        if not st.session_state.extracted_text.strip():
            st.warning("Primero extrae texto con el OCR, o pega texto en el cuadro de arriba.")
        else:
            text = st.session_state.extracted_text.strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "…"

            with st.spinner("Analizando..."):
                if provider == "GROQ":
                    if task == "Resumir en 3 puntos clave":
                        prompt = (
                            "Responde en ESPAÑOL.\n"
                            "Devuelve exactamente 3 viñetas, breves y claras.\n\n"
                            f"Texto:\n{text}\n\n"
                            "Resumen (3 viñetas):\n• "
                        )
                    elif task == "Identificar entidades principales":
                        prompt = (
                            "Responde en ESPAÑOL.\n"
                            "Extrae ENTIDADES con el formato exacto [CATEGORÍA]: valor.\n"
                            "Usa SOLO: PERSONA, ORGANIZACIÓN, LUGAR, FECHA. Una entidad por viñeta.\n\n"
                            f"Texto:\n{text}\n\nRespuesta:\n• "
                        )
                    else:
                        prompt = (
                            "Traduce al inglés el siguiente texto. Mantén nombres propios sin alterar.\n\n"
                            f"Texto: {text}\n\nTraducción:"
                        )

                    out = groq_chat_completion(prompt, model_name=groq_model, temperature=temperature)
                    if out:
                        st.markdown("#### Resultado (GROQ)")
                        st.markdown(f'<div class="result-box">{out}</div>', unsafe_allow_html=True)

                else:
                    if task == "Resumir en 3 puntos clave":
                        summary = hf_summarize(text)
                        if summary:
                            bullets = [s.strip("- •\t ") for s in summary.split("\n") if s.strip()]
                            if len(bullets) >= 3:
                                summary = "\n".join([f"• {b}" for b in bullets[:3]])
                            else:
                                summary = "• " + _clean_ticks(summary)
                            st.markdown("#### Resultado (Hugging Face)")
                            st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)

                    elif task == "Identificar entidades principales":
                        gen = hf_entities(text)
                        if gen:
                            st.markdown("#### Resultado (Hugging Face)")
                            st.markdown(f'<div class="result-box">{_clean_ticks(gen)}</div>', unsafe_allow_html=True)

                    else:
                        translated = hf_translate_to_english(text)
                        if translated:
                            st.markdown("#### Resultado (Hugging Face)")
                            st.markdown(f'<div class="result-box">{translated}</div>', unsafe_allow_html=True)


st.markdown('<hr class="divider" />', unsafe_allow_html=True)
st.caption("Hecho con dedicación para el taller (Streamlit + EasyOCR + GROQ + Hugging Face)")