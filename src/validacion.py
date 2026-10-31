"""Validación aritmética del pedido (Tarea 2, fase 3), sin ground truth.

    suma(cantidad x precio_unitario) - descuento == total_dicho   (con tolerancia)

Si no cuadra, se diagnostica la causa más probable con esta regla, en orden:
 1. ASR: algún precio extraído no coincide con la carta  -> se oyó mal un número.
 2. ASR: la confianza media de Whisper en las palabras del total es baja.
 3. Hablante: la diferencia es exactamente el descuento (promoción no aplicada),
    el subtotal o precio de un ítem (ítem omitido o sobrante), o el total corresponde
    a otra combinación de cantidades del pedido (total del pedido antes de corregir).
 4. ASR: el total y el calculado forman un par de números que suenan parecido (60/70...).
 5. En otro caso, hablante (el ASR estaba seguro de lo que oyó).
"""
from itertools import product as producto_cartesiano

from .carta import id_producto, precio_carta
from .config import cargar_config


def _descuento(promo) -> float:
    if not promo:
        return 0.0
    try:
        return float(promo.get("descuento") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def calcular_total(items) -> float | None:
    try:
        return float(sum(float(i["cantidad"]) * float(i["precio_unitario"]) for i in items))
    except (TypeError, ValueError, KeyError):
        return None


def _otras_cantidades(items, objetivo, tol):
    """¿Existe otra combinación de cantidades (0-3 por ítem) cuya suma da el total dicho?"""
    if len(items) > 6:
        return False
    precios = [float(i["precio_unitario"]) for i in items]
    for cant in producto_cartesiano(range(4), repeat=len(items)):
        if list(cant) != [int(i["cantidad"]) for i in items] and abs(sum(c * p for c, p in zip(cant, precios)) - objetivo) <= tol:
            return True
    return False


def diagnosticar(registro: dict, calculado: float, total: float, confianza_total: float | None) -> tuple[str, str]:
    cfg = cargar_config()["validacion"]
    tol = cfg["tolerancia_soles"]
    items = registro.get("items") or []
    for it in items:
        pc = precio_carta(id_producto(it.get("producto"), it.get("tamano")))
        if pc is not None and it.get("precio_unitario") is not None and abs(float(it["precio_unitario"]) - pc) > tol:
            return "asr", f"precio de {it.get('producto')} ({it['precio_unitario']}) no coincide con la carta ({pc})"
    if confianza_total is not None and confianza_total < cfg["umbral_confianza_total"]:
        return "asr", f"confianza baja del ASR en el total ({confianza_total:.2f})"
    dif = abs(calculado - total)
    desc = _descuento(registro.get("promocion"))
    if desc and abs(dif - desc) <= tol:
        return "hablante", "la diferencia es el descuento: la promoción no se aplicó al total"
    for it in items:
        sub = float(it["cantidad"]) * float(it["precio_unitario"])
        if abs(dif - sub) <= tol or abs(dif - float(it["precio_unitario"])) <= tol:
            return "hablante", f"la diferencia equivale a {it.get('producto')}: ítem omitido o sobrante"
    if _otras_cantidades(items, total + desc, tol):
        return "hablante", "el total corresponde a otras cantidades: probable total del pedido antes de corregir"
    for a, b in cfg["confusiones_asr"]:
        if {round(total), round(calculado)} == {a, b}:
            return "asr", f"{a} y {b} suenan parecido"
    return "hablante", "error de cálculo del hablante (el ASR estaba seguro del total)"


def validar_pedido(registro: dict, confianza_total: float | None = None) -> dict:
    tol = cargar_config()["validacion"]["tolerancia_soles"]
    items, total = registro.get("items"), registro.get("total_dicho")
    suma = calcular_total(items) if items else None
    if suma is None or total is None:
        return {"estado": "no_verificable", "suma_items": suma, "descuento": _descuento(registro.get("promocion")),
                "calculado": None, "total_dicho": total, "diferencia": None, "causa": None, "explicacion":
                "faltan ítems con precio o el total dicho"}
    desc = _descuento(registro.get("promocion"))
    calculado = suma - desc
    try:
        total = float(total)
    except (TypeError, ValueError):
        return {"estado": "no_verificable", "suma_items": suma, "descuento": desc, "calculado": calculado,
                "total_dicho": total, "diferencia": None, "causa": None, "explicacion": "total no numérico"}
    ok = abs(calculado - total) <= tol
    causa, expl = (None, "cuadra") if ok else diagnosticar(registro, calculado, total, confianza_total)
    return {"estado": "cuadra" if ok else "no_cuadra", "suma_items": suma, "descuento": desc,
            "calculado": calculado, "total_dicho": total, "diferencia": round(total - calculado, 2),
            "causa": causa, "explicacion": expl, "confianza_total": confianza_total}


def confianza_del_total(segmentos: list[dict]) -> float | None:
    """Confianza media de Whisper en las palabras numéricas que siguen a la última mención de 'total'."""
    palabras = [w for s in segmentos for w in s.get("palabras", [])]
    idx = [i for i, w in enumerate(palabras) if "total" in w["p"].lower()]
    if not idx:
        return None
    siguientes = palabras[idx[-1] + 1: idx[-1] + 8]
    from .numeros import PALABRAS_NUMERO
    numericas = [w["conf"] for w in siguientes
                 if any(ch.isdigit() for ch in w["p"]) or w["p"].lower().strip(".,") in PALABRAS_NUMERO]
    return round(sum(numericas) / len(numericas), 3) if numericas else None
