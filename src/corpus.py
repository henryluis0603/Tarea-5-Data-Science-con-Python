"""Acceso al corpus: rutas de audio, transcripciones de referencia, ground truth y caché."""
import json
import re
from pathlib import Path

from .config import cargar_config, ruta

EXTENSIONES = (".m4a", ".wav", ".mp3", ".ogg", ".opus")


def ruta_audio(clip: str) -> Path | None:
    for ext in EXTENSIONES:
        p = ruta("audio") / f"{clip}{ext}"
        if p.exists():
            return p
    return None


def referencia(clip: str) -> str:
    """Transcripción de referencia = las líneas habladas del guion (lo que se leyó)."""
    carpeta = ruta("guiones_entrevistas") if clip.startswith("E") else ruta("guiones_llamadas")
    lineas = (carpeta / f"{clip}.md").read_text(encoding="utf-8").splitlines()
    return " ".join(re.sub(r"^\*\*[AB]:\*\*\s*", "", l) for l in lineas if re.match(r"^\*\*[AB]:\*\*", l))


def ground_truth(tarea: str) -> dict:
    clave = "ground_truth_entrevistas" if tarea == "entrevista" else "ground_truth_llamadas"
    datos = json.loads(ruta(clave).read_text(encoding="utf-8"))
    return {c["clip_id"]: c for c in datos["clips"]}


def clips(tarea: str) -> list[str]:
    return sorted(ground_truth(tarea))


def ruta_cache(clip: str, variante: str) -> Path:
    return ruta("transcripciones") / variante / f"{clip}.json"


def leer_cache(clip: str, variante: str) -> dict | None:
    p = ruta_cache(clip, variante)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def guardar_cache(clip: str, variante: str, datos: dict) -> None:
    p = ruta_cache(clip, variante)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def variante_base() -> str:
    a = cargar_config()["asr"]
    return f"{a['modelo']}_{a['precision']}"
