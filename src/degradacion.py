"""Escalera de degradación reproducible (Tarea 2, fase 2).

L0  tal como se grabó.
L1  nota de voz de WhatsApp: mono, Opus a baja tasa de bits, 16 kHz.
L2  llamada telefónica: pasa-banda 300-3400 Hz (Butterworth) y remuestreo a 8 kHz.
L3  ambiente ruidoso: L1 + ruido de conversación de fondo (babble) a una SNR fija.

La SNR se MIDE, no se asume: potencia de la voz en los tramos activos (energía por encima
del percentil 30 en ventanas de 20 ms) contra la potencia del ruido escalado.
"""
import io

import av
import numpy as np
from faster_whisper.audio import decode_audio
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import cargar_config

SR = 16000


def _codificar_opus(x: np.ndarray, bitrate: int, sr: int) -> np.ndarray:
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as cont:
        st = cont.add_stream("libopus", rate=sr)
        st.bit_rate = bitrate
        st.layout = "mono"
        frame = av.AudioFrame.from_ndarray((np.clip(x, -1, 1) * 32767).astype(np.int16)[None, :],
                                           format="s16", layout="mono")
        frame.sample_rate = sr
        for p in st.encode(frame):
            cont.mux(p)
        for p in st.encode(None):
            cont.mux(p)
    buf.seek(0)
    return decode_audio(buf, sampling_rate=SR)


def _potencia_activa(x: np.ndarray) -> float:
    v = 320  # 20 ms a 16 kHz
    marcos = x[: len(x) // v * v].reshape(-1, v)
    energia = (marcos ** 2).mean(axis=1)
    return float(energia[energia > np.percentile(energia, 30)].mean())


def snr_medida(voz: np.ndarray, ruido: np.ndarray) -> float:
    return 10 * np.log10(_potencia_activa(voz) / float((ruido ** 2).mean()))


def ruido_babble(longitud: int, fuentes: list[str], voces: int, semilla: int) -> np.ndarray:
    """Suma de `voces` clips (de las ENTREVISTAS, no de las llamadas) con desfases aleatorios."""
    rng = np.random.default_rng(semilla)
    total = np.zeros(longitud)
    for ruta in rng.choice(fuentes, size=voces, replace=False):
        s = decode_audio(str(ruta), sampling_rate=SR)
        s = np.tile(s, int(np.ceil(longitud / len(s))) + 1)
        ini = rng.integers(0, len(s) - longitud)
        total += s[ini: ini + longitud] / (np.abs(s).max() + 1e-9)
    return total


def degradar(ruta: str, nivel: str, fuentes_babble: list[str]) -> tuple[np.ndarray, dict]:
    cfg = cargar_config()["degradacion"]
    x = decode_audio(ruta, sampling_rate=SR)
    if nivel == "L0":
        return x, {}
    if nivel == "L1":
        c = cfg["L1"]
        return _codificar_opus(x, c["bitrate"], c["frecuencia"]), {"bitrate": c["bitrate"]}
    if nivel == "L2":
        c = cfg["L2"]
        sos = butter(c["orden_filtro"], c["banda_hz"], btype="bandpass", fs=SR, output="sos")
        y = resample_poly(sosfiltfilt(sos, x), c["frecuencia"], SR)   # a 8 kHz
        return resample_poly(y, SR, c["frecuencia"]).astype(np.float32), {"banda_hz": c["banda_hz"]}
    if nivel == "L3":
        c = cfg["L3"]
        base, _ = degradar(ruta, c["base"], fuentes_babble)
        ruido = ruido_babble(len(base), fuentes_babble, c["voces_babble"], cfg["semilla"])
        escala = np.sqrt(_potencia_activa(base) / ((ruido ** 2).mean() * 10 ** (c["snr_db"] / 10)))
        ruido = ruido * escala
        y = base + ruido
        y = y / max(1.0, np.abs(y).max())
        snr_global = 10 * np.log10(float((base ** 2).mean()) / float((ruido ** 2).mean()))
        return y.astype(np.float32), {"snr_objetivo_db": c["snr_db"],
                                      "snr_medida_voz_activa_db": round(float(snr_medida(base, ruido)), 2),
                                      "snr_medida_global_db": round(float(snr_global), 2)}
    raise ValueError(nivel)
