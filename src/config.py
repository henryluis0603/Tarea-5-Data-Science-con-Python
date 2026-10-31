"""Carga de config.yaml y .env. Todas las rutas se resuelven desde la raíz del proyecto."""
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def cargar_config(ruta: str = "config.yaml") -> dict:
    load_dotenv(RAIZ / ".env")
    with open(RAIZ / ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(clave: str) -> Path:
    """Ruta absoluta para una clave de la sección `rutas`."""
    return RAIZ / cargar_config()["rutas"][clave]
