"""App de Streamlit: carga resultados precalculados; solo transcribe en el probador.

    streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import cargar_config, ruta
from src.corpus import clips, ground_truth, leer_cache, referencia, ruta_audio, variante_base
from src.engine import procesar_audio

st.set_page_config(page_title="De la voz al registro", layout="wide")
CFG = cargar_config()
RES_PATH = ruta("resultados") / "resultados.json"
GPU_PATH = ruta("benchmark_gpu")


@st.cache_data
def cargar_resultados():
    return json.loads(RES_PATH.read_text(encoding="utf-8")) if RES_PATH.exists() else None


@st.cache_data
def cargar_gpu():
    return json.loads(GPU_PATH.read_text(encoding="utf-8")) if GPU_PATH.exists() else None


def registro_guardado(clip, nivel, metodo):
    p = ruta("resultados") / "registros" / variante_base() / nivel / metodo / f"{clip}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tabla_verificacion(r):
    filas = [{"campo": c, "valor": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
              "verificación": r["verificacion"].get(c), "evidencia": r["evidencias"].get(c)}
             for c, v in r["campos"].items()]
    return pd.DataFrame(filas)


RES, GPU = cargar_resultados(), cargar_gpu()
st.title("De la voz al registro")
st.caption("Entrevistas y llamadas de venta → registros verificados. Todos los datos son inventados.")

tabs = st.tabs(["1 · Problema y método", "2 · Corpus", "3 · Resultados", "4 · Pedidos y escalera", "5 · Probador"])

# ------------------------------------------------------------------ 1
with tabs[0]:
    st.header("El audio que nadie procesa")
    st.markdown("""
Transcribir ya lo hace el celular. Lo que falta es pasar de **texto** a **campos verificados en una tabla**,
con una medida de cuánto confiar en cada uno.

**Pipeline (un solo motor, `src/engine.py::procesar_audio`)**

1. **ASR** — faster-whisper; el dispositivo se lee del modelo cargado y la VRAM se mide con NVML.
2. **Extracción** — dos caminos comparables: reglas (regex) y LLM con JSON (OpenAI `gpt-4o-mini` y Ollama `qwen2.5:3b` local).
3. **Verificación literal** — cada valor debe existir en la transcripción tras normalizar
   números dichos en palabras ("seis mil quinientos" → 6500). Si no, `sin_verificar`.
4. **Validación aritmética** (solo llamadas) — Σ cantidad × precio − promoción = total dicho.
   Si no cuadra, se diagnostica si fue el ASR o el hablante.

**Corre una vez:** guiones → grabación → transcripción a caché → evaluación.
**Corre por cada audio nuevo:** transcripción → extracción → verificación → validación.
""")
    st.info("Este sistema **no** debe usarse para decidir contrataciones ni cobros sin revisión humana: "
            "con 5 clips por tarea, los intervalos de confianza son anchos.")

# ------------------------------------------------------------------ 2
with tabs[1]:
    tarea = st.radio("Corpus", ["entrevista", "llamada"], horizontal=True)
    clip = st.selectbox("Clip", clips(tarea))
    gt = ground_truth(tarea)[clip]
    audio = ruta_audio(clip)
    if audio:
        st.audio(str(audio))
    else:
        st.warning("El audio no está en el repositorio (ver README para solicitarlo).")
    tr = leer_cache(clip, f"{variante_base()}/L0")
    c1, c2 = st.columns(2)
    c1.subheader("Referencia (guion leído)")
    c1.write(referencia(clip))
    c2.subheader(f"ASR ({variante_base()})")
    c2.write(tr["texto"] if tr else "Sin transcripción en caché.")
    st.subheader("Ground truth")
    st.json({k: v for k, v in gt.items() if k not in ("rol_A", "rol_B")})

# ------------------------------------------------------------------ 3
with tabs[2]:
    if not RES:
        st.warning("Falta results/resultados.json: corre `python -m scripts.evaluar`.")
    else:
        t1 = RES["tarea1"]
        st.header("Capa 1 — Transcripción")
        st.dataframe(pd.DataFrame(t1["transcripcion"]).round(3), hide_index=True)
        if GPU:
            st.subheader("GPU: modelos, precisión y dispositivo (Colab)")
            st.caption(f"GPU: {GPU['gpu']}")
            g = pd.DataFrame(GPU["filas"]).drop(columns=["texto"])
            st.dataframe(g.groupby(["modelo", "dispositivo_pedido", "dispositivo_real", "precision", "modo_glosario"])
                         [["wer", "cer", "seg_por_min_audio", "vram_pico_mb", "vram_modelo_mb"]].mean().round(3))
        st.subheader("Glosario de dominio")
        st.dataframe(pd.DataFrame(RES["glosario"]["comparaciones"]), hide_index=True)

        st.header("Capa 2 — Registro")
        campos = pd.DataFrame(t1["campos"])
        campos["IC 95%"] = campos.apply(lambda r: f"[{r.ic95_inf:.2f}, {r.ic95_sup:.2f}]", axis=1)
        st.dataframe(campos.pivot(index="campo", columns="metodo", values="exactitud").round(2))
        st.caption("Exactitud por campo (5 entrevistas). IC de Wilson en results/t1_campos.csv.")
        st.write("**F1 macro (nivel de inglés):**", {k: round(v, 3) for k, v in t1["f1_macro_nivel_ingles"].items()})
        st.write("**Error absoluto medio:**")
        st.dataframe(pd.DataFrame(t1["mae"]).round(2), hide_index=True)

        st.subheader("¿Predice el WER si el registro sirve?")
        pts = pd.DataFrame(RES["wer_vs_registro"])
        st.altair_chart(alt.Chart(pts).mark_circle(size=80).encode(
            x=alt.X("wer:Q", title="WER"), y=alt.Y("exactitud_registro:Q", title="Exactitud del registro"),
            color="metodo:N", shape="tarea:N", tooltip=["tarea", "clip", "nivel", "metodo", "wer", "exactitud_registro"],
        ).properties(height=380), width="stretch")

# ------------------------------------------------------------------ 4
with tabs[3]:
    if not RES:
        st.warning("Falta results/resultados.json.")
    else:
        t2 = RES["tarea2"]
        esc = pd.DataFrame(t2["escalera"])
        st.header("Escalera de degradación")
        st.markdown("L0 tal cual · L1 nota de voz (Opus 12 kbps) · L2 teléfono (300–3400 Hz, 8 kHz) · "
                    f"L3 L1 + ruido de conversación a {CFG['degradacion']['L3']['snr_db']} dB SNR")
        largo = esc.melt(id_vars=["nivel", "metodo"], value_vars=["wer_medio", "exactitud_campos"],
                         var_name="métrica", value_name="valor")
        st.altair_chart(alt.Chart(largo).mark_line(point=True).encode(
            x="nivel:N", y="valor:Q", color="metodo:N", strokeDash="métrica:N",
            tooltip=["nivel", "metodo", "métrica", "valor"]).properties(height=360), width="stretch")
        st.dataframe(esc.round(3), hide_index=True)
        if t2["snr"]:
            st.caption("SNR medida en L3")
            st.dataframe(pd.DataFrame(t2["snr"]), hide_index=True)

        st.header("Validación aritmética: detectados vs omitidos")
        st.dataframe(esc[["nivel", "metodo", "trampas", "detectados", "omitidos", "falsas_alarmas",
                          "no_verificables", "diagnosticos_evaluados", "diagnosticos_correctos"]], hide_index=True)

        st.header("Ver un pedido")
        c1, c2, c3 = st.columns(3)
        clip = c1.selectbox("Llamada", clips("llamada"))
        nivel = c2.selectbox("Nivel", CFG["degradacion"]["niveles"])
        metodo = c3.selectbox("Método", sorted(esc["metodo"].unique()))
        r = registro_guardado(clip, nivel, metodo)
        if r:
            v = r["validacion"] or {}
            color = {"cuadra": "green", "no_cuadra": "red"}.get(v.get("estado"), "gray")
            st.markdown(f"**Validación:** :{color}[{v.get('estado')}] — calculado {v.get('calculado')} "
                        f"vs dicho {v.get('total_dicho')}. **Causa:** {v.get('causa')} — {v.get('explicacion')}")
            st.dataframe(tabla_verificacion(r), hide_index=True)
            with st.expander("Transcripción"):
                st.write(r["transcripcion"])

# ------------------------------------------------------------------ 5
with tabs[4]:
    st.header("Probador")
    st.caption("Sube un audio: se transcribe aquí mismo con el motor (puede tardar ~1 min por cada 2 min de audio en CPU).")
    archivo = st.file_uploader("Audio", type=["m4a", "mp3", "wav", "ogg", "opus"])
    c1, c2 = st.columns(2)
    esquema = c1.selectbox("Esquema", ["entrevista", "llamada"])
    opcion = c2.selectbox("Extracción", ["llm · openai", "llm · ollama", "reglas"])
    if archivo and st.button("Procesar"):
        with tempfile.NamedTemporaryFile(suffix=Path(archivo.name).suffix, delete=False) as f:
            f.write(archivo.read())
        metodo, _, proveedor = opcion.partition(" · ")
        with st.spinner("Transcribiendo y extrayendo..."):
            r = procesar_audio(f.name, esquema, metodo, proveedor or None)
        if r["error"]:
            st.error(r["error"])
        st.write(f"Dispositivo: **{r['dispositivo']}** · ASR {r['modelo_asr']} · "
                 f"{r['segundos']['asr']} s ASR + {r['segundos']['extraccion']} s extracción")
        st.subheader("Transcripción")
        st.write(r["transcripcion"])
        st.subheader("Registro")
        st.dataframe(tabla_verificacion(r), hide_index=True)
        if r["validacion"]:
            st.subheader("Validación aritmética")
            st.json(r["validacion"])
