"""Regla de verificación: todo valor extraído debe existir LITERALMENTE en la transcripción.

Se compara después de `normalizar_para_verificar` (mismo tratamiento para valor y transcripción),
que resuelve el caso propio del audio: "seis mil quinientos" en el texto contra 6500 en el
registro. Si el valor no aparece, el campo queda `sin_verificar`, aunque sea correcto: un modelo
que corrige en silencio lo que el ASR oyó mal es tan peligroso como uno que inventa.
"""
from datetime import date

from .carta import id_producto
from .config import cargar_config
from .normalizacion import MESES, contiene, normalizar_para_verificar, normalizar_texto

VERIFICADO, SIN_VERIFICAR, VACIO = "verificado", "sin_verificar", "vacio"
NOMBRE_MES = MESES.split("|")


def _num(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else str(v)


def _alias(seccion: str, clave: str) -> list[str]:
    return [normalizar_texto(a) for a in cargar_config()["catalogo"][seccion].get(clave, [])] + [normalizar_texto(clave)]


def _formas_disponibilidad(v: dict) -> list[str]:
    tipo = v.get("tipo")
    if tipo == "inmediata":
        return ["inmediata", "inmediato", "de inmediato"]
    if tipo == "relativa_dias" and v.get("dias"):
        n = int(v["dias"])
        formas = [f"{n} dias"]
        if n % 7 == 0:
            formas += [f"{n // 7} semana", f"{n // 7} semanas"] + (["proxima semana"] if n == 7 else [])
        if n % 30 == 0:
            formas += [f"{n // 30} mes", f"{n // 30} meses"]
        return formas
    if tipo == "fecha" and v.get("fecha"):
        d = date.fromisoformat(v["fecha"])
        return [f"{d.day} de {NOMBRE_MES[d.month - 1]}"]
    return []


def _item_ok(t: str, it: dict) -> bool:
    pid = id_producto(it.get("producto"), it.get("tamano"))
    nombres = [normalizar_texto(it.get("producto") or "")]
    if pid:
        nombres += [normalizar_texto(a) for a in cargar_config()["catalogo"]["carta"][pid]["alias"]]
    nombre_ok = any(contiene(t, n) for n in nombres if n)
    precio_ok = it.get("precio_unitario") is not None and contiene(t, _num(it["precio_unitario"]))
    cant = it.get("cantidad")
    cantidad_ok = cant is not None and (int(cant) == 1 or contiene(t, _num(cant)))
    return nombre_ok and precio_ok and cantidad_ok


def verificar_campo(t: str, campo: str, valor) -> str:
    if valor in (None, "", []):
        return VACIO
    try:
        if campo in ("anios_experiencia", "total_dicho"):
            ok = contiene(t, _num(valor))
        elif campo == "pretension_salarial":
            ok = contiene(t, _num(valor["monto"]))
        elif campo in ("ultimo_puesto",):
            ok = contiene(t, normalizar_para_verificar(valor))
        elif campo == "tecnologias":
            ok = all(any(contiene(t, a) for a in _alias("tecnologias", x)) for x in valor)
        elif campo == "disponibilidad":
            ok = any(contiene(t, f) for f in _formas_disponibilidad(valor))
        elif campo == "nivel_ingles":
            ok = any(contiene(t, a) for a in _alias("niveles_ingles", valor))
        elif campo == "metodo_pago":
            ok = any(contiene(t, a) for a in _alias("metodos_pago", valor))
        elif campo == "items":
            ok = all(_item_ok(t, it) for it in valor)
        elif campo == "promocion":
            nombre = normalizar_texto(valor.get("nombre") or "")
            alias = [nombre] + [normalizar_texto(a) for p in cargar_config()["catalogo"]["promociones"].values()
                                if normalizar_texto(p["nombre"]) == nombre or nombre in normalizar_texto(" ".join(p["alias"]))
                                for a in p["alias"]]
            ok = any(contiene(t, a) for a in alias if a) and contiene(t, _num(valor.get("descuento") or 0))
        elif campo == "direccion_entrega":
            partes = [normalizar_para_verificar(p) for p in str(valor).split(",")]
            ok = all(contiene(t, p) for p in partes if p)
        else:
            ok = contiene(t, normalizar_para_verificar(str(valor)))
    except (TypeError, ValueError, KeyError, AttributeError):
        ok = False
    return VERIFICADO if ok else SIN_VERIFICAR


def verificar(transcripcion: str, registro: dict) -> dict:
    t = normalizar_para_verificar(transcripcion)
    return {campo: verificar_campo(t, campo, valor) for campo, valor in registro.items()}
