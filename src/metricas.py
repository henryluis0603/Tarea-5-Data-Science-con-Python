"""Métricas de las dos capas: transcripción (WER, CER) y registro (exactitud por campo,
F1 macro del campo categórico, error absoluto de los numéricos), con intervalos de confianza."""
from datetime import date, timedelta
from difflib import SequenceMatcher

import jiwer
import numpy as np
from sklearn.metrics import f1_score

from .carta import id_producto
from .config import cargar_config
from .normalizacion import normalizar_para_verificar, normalizar_texto


# ------------------------------------------------------------ capa transcripción

def wer_cer(referencia: str, hipotesis: str) -> dict:
    r, h = normalizar_texto(referencia), normalizar_texto(hipotesis)
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h)}


# ------------------------------------------------------------ intervalos

def wilson(aciertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = aciertos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    radio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centro - radio), min(1.0, centro + radio))


def bootstrap_media(valores, n: int = 10000, semilla: int = 42) -> tuple[float, float]:
    v = np.asarray(valores, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))


# ------------------------------------------------------------ capa registro: entrevista

def _canon_tec(nombre: str) -> str:
    cat = cargar_config()["catalogo"]["tecnologias"]
    n = normalizar_texto(nombre)
    for canon, alias in cat.items():
        if n == normalizar_texto(canon) or n in {normalizar_texto(a) for a in alias}:
            return canon
    return n


def fecha_disponible(v) -> str | None:
    if not isinstance(v, dict):
        return None
    ref = date.fromisoformat(cargar_config()["proyecto"]["fecha_grabacion"])
    if v.get("tipo") == "inmediata":
        return ref.isoformat()
    if v.get("tipo") == "relativa_dias" and v.get("dias") is not None:
        return (ref + timedelta(days=int(v["dias"]))).isoformat()
    if v.get("tipo") == "fecha" and v.get("fecha"):
        return str(v["fecha"])
    return None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def similitud(a, b) -> float:
    return SequenceMatcher(None, normalizar_para_verificar(str(a or "")), normalizar_para_verificar(str(b or ""))).ratio()


def puntuar_entrevista(pred: dict, gt: dict) -> dict:
    umbral = cargar_config()["verificacion"]["similitud_texto_min"]
    p_anios, g_anios = _num(pred.get("anios_experiencia")), gt["anios_experiencia"]
    p_sal = pred.get("pretension_salarial") or {}
    p_monto = _num(p_sal.get("monto")) if isinstance(p_sal, dict) else None
    g_sal = gt["pretension_salarial"]
    p_tec = {_canon_tec(x) for x in (pred.get("tecnologias") or []) if isinstance(x, str)}
    g_tec = {_canon_tec(x) for x in gt["tecnologias"]}
    inter = len(p_tec & g_tec)
    f1_tec = 2 * inter / (len(p_tec) + len(g_tec)) if (p_tec or g_tec) else 1.0
    return {
        "anios_experiencia": {"correcto": p_anios is not None and abs(p_anios - g_anios) < 0.01,
                              "error_abs": abs(p_anios - g_anios) if p_anios is not None else None},
        "ultimo_puesto": {"correcto": similitud(pred.get("ultimo_puesto"), gt["ultimo_puesto"]) >= umbral},
        "tecnologias": {"correcto": p_tec == g_tec, "f1": f1_tec},
        "pretension_salarial": {
            "correcto": p_monto is not None and abs(p_monto - g_sal["monto"]) < 0.5
                        and (p_sal.get("moneda") or "PEN").upper() == g_sal["moneda"],
            "error_abs": abs(p_monto - g_sal["monto"]) if p_monto is not None else None},
        "disponibilidad": {"correcto": fecha_disponible(pred.get("disponibilidad")) == fecha_disponible(gt["disponibilidad"])},
        "nivel_ingles": {"correcto": normalizar_texto(str(pred.get("nivel_ingles") or "")) == gt["nivel_ingles"]},
    }


def f1_macro(verdaderos: list[str], predichos: list[str]) -> float:
    pred = [normalizar_texto(str(p or "ninguno")) for p in predichos]
    return float(f1_score(verdaderos, pred, labels=sorted(set(verdaderos)), average="macro", zero_division=0))


# ------------------------------------------------------------ capa registro: llamada

def items_por_producto(items) -> dict:
    """Agrega por id de carta: {id: (cantidad_total, precio_unitario)}. Los sabores de pizza se ignoran."""
    agg = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        pid = id_producto(it.get("producto"), it.get("tamano")) or normalizar_texto(str(it.get("producto")))
        cant, precio = _num(it.get("cantidad")) or 0, _num(it.get("precio_unitario"))
        c0, _ = agg.get(pid, (0, precio))
        agg[pid] = (c0 + cant, precio)
    return agg


def puntuar_llamada(pred: dict, gt: dict) -> dict:
    umbral = cargar_config()["verificacion"]["similitud_texto_min"]
    p_items, g_items = items_por_producto(pred.get("items")), items_por_producto(gt["items"])
    p_set, g_set = set(p_items.items()), set(g_items.items())
    f1 = 2 * len(p_set & g_set) / (len(p_set) + len(g_set)) if (p_set or g_set) else 1.0
    p_promo = pred.get("promocion")
    p_desc = _num(p_promo.get("descuento")) if isinstance(p_promo, dict) else 0.0
    g_desc = gt["promocion"]["descuento"] if gt["promocion"] else 0.0
    p_total = _num(pred.get("total_dicho"))
    return {
        "items": {"correcto": p_items == g_items, "f1": f1},
        "promocion": {"correcto": abs((p_desc or 0) - g_desc) < 0.5},
        "direccion_entrega": {"correcto": similitud(pred.get("direccion_entrega"), gt["direccion_entrega"]) >= umbral},
        "metodo_pago": {"correcto": normalizar_texto(str(pred.get("metodo_pago") or "")) == gt["metodo_pago"]},
        "total_dicho": {"correcto": p_total is not None and abs(p_total - gt["total_dicho"]) < 0.5,
                        "error_abs": abs(p_total - gt["total_dicho"]) if p_total is not None else None},
    }
