"""Transcripción con faster-whisper.

El dispositivo que se reporta se lee del modelo cargado en tiempo de ejecución
(`WhisperModel.model.device`), no de config.yaml. Con GPU NVIDIA se mide además la VRAM
pico con NVML mientras transcribe: es la evidencia de que la GPU se usó de verdad.
"""
import threading
import time

import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from .config import cargar_config

_MODELOS: dict = {}


def _precargar_cuda12():
    """CTranslate2 4.x busca cuBLAS/cuDNN de CUDA 12. En entornos con CUDA 13 (Colab desde 2026)
    no están en el sistema: se instalan por pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) y se
    cargan aquí con RTLD_GLOBAL, porque LD_LIBRARY_PATH solo se lee al iniciar el proceso."""
    import ctypes
    import glob
    import os
    for modulo in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            carpeta = list(__import__(modulo, fromlist=["_"]).__path__)[0]
        except ImportError:
            continue
        for so in sorted(glob.glob(os.path.join(carpeta, "lib*.so*")), key=lambda s: ("Lt" not in s, s)):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_precargar_cuda12()


class MonitorVRAM:
    """Muestrea la memoria usada de la GPU 0 en un hilo y guarda el pico (MB)."""

    def __init__(self, intervalo: float):
        self.intervalo, self.pico_mb, self.base_mb = intervalo, None, None
        self._parar = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _usada(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20

    def _bucle(self):
        while not self._parar.is_set():
            self.pico_mb = max(self.pico_mb or 0, self._usada())
            time.sleep(self.intervalo)

    def __enter__(self):
        if self._nvml:
            self.base_mb = self._usada()
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()
        return self

    def __exit__(self, *exc):
        if self._nvml:
            self._parar.set()
            self._hilo.join()


def gpus_disponibles() -> int:
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def cargar_modelo(nombre: str, dispositivo: str, precision: str) -> WhisperModel:
    if dispositivo == "auto":
        dispositivo = "cuda" if gpus_disponibles() > 0 else "cpu"
    clave = (nombre, dispositivo, precision)
    if clave not in _MODELOS:
        _MODELOS[clave] = WhisperModel(nombre, device=dispositivo, compute_type=precision)
    return _MODELOS[clave]


def transcribir(audio, modelo: str | None = None, dispositivo: str | None = None,
                precision: str | None = None, modo_glosario: str | None = None) -> dict:
    """Transcribe una ruta de audio (o un array a 16 kHz). Devuelve texto, segmentos con
    confianza por palabra, dispositivo real, tiempos y VRAM pico."""
    cfg = cargar_config()["asr"]
    modelo = modelo or cfg["modelo"]
    dispositivo = dispositivo or cfg["dispositivo"]
    precision = precision or cfg["precision"]
    modo_glosario = modo_glosario or cfg["modo_glosario"]

    t0 = time.perf_counter()
    m = cargar_modelo(modelo, dispositivo, precision)
    carga_s = time.perf_counter() - t0

    señal = decode_audio(audio) if isinstance(audio, str) else audio
    duracion_s = len(señal) / 16000
    extra = {}
    if modo_glosario == "hotwords":
        extra["hotwords"] = cfg["glosario"]
    elif modo_glosario == "initial_prompt":
        extra["initial_prompt"] = cfg["glosario"]

    intervalo = cargar_config()["benchmark"]["intervalo_vram_s"]
    with MonitorVRAM(intervalo) as vram:
        t1 = time.perf_counter()
        segmentos, info = m.transcribe(señal, language=cfg["idioma"], beam_size=cfg["beam_size"],
                                       word_timestamps=cfg["timestamps_palabra"], **extra)
        segs = [{
            "inicio": round(s.start, 2), "fin": round(s.end, 2), "texto": s.text.strip(),
            "palabras": [{"p": w.word.strip(), "conf": round(w.probability, 3)} for w in (s.words or [])],
        } for s in segmentos]   # el generador se consume aquí: la transcripción ocurre dentro del monitor
        segundos = time.perf_counter() - t1

    return {
        "texto": " ".join(s["texto"] for s in segs),
        "segmentos": segs,
        "modelo": modelo,
        "precision": precision,
        "modo_glosario": modo_glosario,
        "dispositivo_real": str(m.model.device),
        "gpus_visibles": gpus_disponibles(),
        "duracion_audio_s": round(duracion_s, 2),
        "segundos_transcripcion": round(segundos, 2),
        "segundos_carga_modelo": round(carga_s, 2),
        "seg_por_min_audio": round(segundos / (duracion_s / 60), 2),
        "vram_base_mb": vram.base_mb,
        "vram_pico_mb": vram.pico_mb,
        "idioma_detectado": info.language,
        "faster_whisper": __import__("faster_whisper").__version__,
        "ctranslate2": ctranslate2.__version__,
    }
