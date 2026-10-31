"""Mapeo de un producto dicho (nombre libre + tamaño) a su id en la carta de config.yaml."""
from .config import cargar_config
from .normalizacion import normalizar_texto

TAMANOS_PIZZA = {"personal", "mediana", "familiar"}
TAMANOS_POLLO = {"cuarto": "pollo_cuarto", "medio": "pollo_medio", "entero": "pollo_entero"}


def id_producto(producto: str | None, tamano: str | None = None) -> str | None:
    carta = cargar_config()["catalogo"]["carta"]
    p = normalizar_texto(producto or "")
    tam = normalizar_texto(tamano or "")
    texto = f"{p} {tam}"
    if "pizza" in texto or any(t in texto.split() for t in TAMANOS_PIZZA):
        for t in TAMANOS_PIZZA:
            if t in texto.split():
                return f"pizza_{t}"
    if "pollo" in texto:
        for t, pid in TAMANOS_POLLO.items():
            if t in texto.split():
                return pid
    mejor = None
    for pid, d in carta.items():
        for a in [d["nombre"], *d["alias"]]:
            a = normalizar_texto(a)
            if a and a in p and (mejor is None or len(a) > mejor[1]):
                mejor = (pid, len(a))
    return mejor[0] if mejor else None


def precio_carta(pid: str | None) -> float | None:
    return cargar_config()["catalogo"]["carta"].get(pid, {}).get("precio") if pid else None
