"""Camino 2: un modelo de lenguaje devuelve el registro como JSON estructurado.

Dos proveedores con el mismo prompt (config.yaml): OpenAI (API) y Ollama (modelo local).
"""
import json
import os

import httpx

from ..config import cargar_config


def _mensajes(transcripcion: str, clave_prompt: str) -> list[dict]:
    cfg = cargar_config()["extraccion"]
    return [
        {"role": "system", "content": cfg["prompt_sistema"]},
        {"role": "user", "content": cfg[clave_prompt].replace("{transcripcion}", transcripcion)},
    ]


def _openai(mensajes) -> str:
    from openai import OpenAI
    cfg = cargar_config()["extraccion"]["openai"]
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY en .env")
    r = OpenAI().chat.completions.create(model=cfg["modelo"], temperature=cfg["temperatura"],
                                         messages=mensajes, response_format={"type": "json_object"})
    return r.choices[0].message.content


def _ollama(mensajes) -> str:
    cfg = cargar_config()["extraccion"]["ollama"]
    r = httpx.post(f"{cfg['url']}/api/chat", timeout=cfg["timeout_s"], json={
        "model": cfg["modelo"], "messages": mensajes, "stream": False, "format": "json",
        "options": {"temperature": cfg["temperatura"]},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def nombre_modelo(proveedor: str) -> str:
    return cargar_config()["extraccion"][proveedor]["modelo"]


def extraer(transcripcion: str, clave_prompt: str, campos: list[str], proveedor: str) -> dict:
    crudo = {"openai": _openai, "ollama": _ollama}[proveedor](_mensajes(transcripcion, clave_prompt))
    datos = json.loads(crudo)
    salida = {}
    for c in campos:
        v = datos.get(c)
        if isinstance(v, dict) and "valor" in v:
            salida[c] = {"valor": v.get("valor"), "evidencia": v.get("evidencia")}
        else:   # modelos chicos a veces omiten la envoltura {"valor", "evidencia"}
            salida[c] = {"valor": v, "evidencia": None}
    return salida
