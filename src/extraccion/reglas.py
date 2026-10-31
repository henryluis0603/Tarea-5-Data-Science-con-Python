"""Camino 1: extracción por reglas (expresiones regulares sobre el texto normalizado).

Todas las reglas trabajan sobre `normalizar_para_verificar(transcripcion)`: minúsculas, sin
tildes ni puntuación y con los números en dígitos. La evidencia que devuelven es el tramo
normalizado que hizo match, así que siempre existe literalmente en la transcripción.
"""
import re
from datetime import date, timedelta

from ..config import cargar_config
from ..normalizacion import MESES, contiene, normalizar_para_verificar, normalizar_texto

NUM_MES = {m: i + 1 for i, m in enumerate(MESES.replace("|setiembre", "").split("|"))} | {"setiembre": 9}


def _campo(valor, evidencia):
    return {"valor": valor, "evidencia": evidencia}


def _alias_norm(lista):
    return [normalizar_texto(a) for a in lista]


# ---------------------------------------------------------------- Entrevista

def _anios(t):
    hallados = [(float(m.group(1)), m.group(0)) for m in re.finditer(r"\b(\d+(?:\.\d+)?) años?\b", t)]
    hallados = [h for h in hallados if h[0] <= 50]
    if not hallados:
        return _campo(None, None)
    v, ev = max(hallados)          # la cifra mayor suele ser el total
    return _campo(int(v) if v.is_integer() else v, ev)


def _puesto(t):
    patrones = [r"mi ultimo puesto fue", r"mi (?:cargo|puesto)(?: actual)? es", r"trabajo como", r"soy(?! del?\b)"]
    corte = r"(?= en | coordino| me | construyo| armamos| ahi|$)"
    for p in patrones:
        m = re.search(rf"\b{p} (.+?){corte}", t)
        if m:
            valor = " ".join(m.group(1).split()[:6])
            return _campo(valor, m.group(0))
    return _campo(None, None)


def _tecnologias(t):
    cat = cargar_config()["catalogo"]["tecnologias"]
    encontradas, evid = [], []
    for nombre, alias in cat.items():
        for a in sorted(_alias_norm(alias), key=len, reverse=True):
            if contiene(t, a):
                encontradas.append(nombre)
                evid.append(a)
                break
    return _campo(encontradas or None, " | ".join(evid) or None)


def _salario(t):
    candidatos = []
    for m in re.finditer(r"\b(\d+(?:\.\d+)?) (soles|dolares)\b", t):
        previo = t[max(0, m.start() - 30): m.start()]
        if re.search(r"\bgan(o|as|a)\b", previo):   # lo que gana hoy no es la pretensión
            continue
        candidatos.append(m)
    if not candidatos:
        return _campo(None, None)
    m = candidatos[-1]             # vale la última cifra (cubre autocorrecciones)
    monto = float(m.group(1))
    return _campo({"monto": int(monto) if monto.is_integer() else monto,
                   "moneda": "USD" if m.group(2) == "dolares" else "PEN"}, m.group(0))


def _disponibilidad(t):
    reglas = [
        (rf"\b(\d+) de ({MESES})\b", lambda m: {"tipo": "fecha", "dia": int(m.group(1)), "mes": NUM_MES[m.group(2)]}),
        (r"\b(\d+) dias\b(?! a la semana)", lambda m: {"tipo": "relativa_dias", "dias": int(m.group(1))}),
        (r"\ben (\d+) semanas?\b", lambda m: {"tipo": "relativa_dias", "dias": 7 * int(m.group(1))}),
        (r"\bla proxima semana\b", lambda m: {"tipo": "relativa_dias", "dias": 7}),
        (r"\b(\d+) mes(?:es)? de preaviso\b", lambda m: {"tipo": "relativa_dias", "dias": 30 * int(m.group(1))}),
        (r"\binmediat[ao]\b|\bde inmediato\b", lambda m: {"tipo": "inmediata"}),
    ]
    for patron, f in reglas:
        m = re.search(patron, t)
        if m:
            v = f(m)
            if v["tipo"] == "fecha":
                v = {"tipo": "fecha", "fecha": _fecha_futura(v["dia"], v["mes"])}
            return _campo(v, m.group(0))
    return _campo(None, None)


def _fecha_futura(dia, mes):
    ref = date.fromisoformat(cargar_config()["proyecto"]["fecha_grabacion"])
    d = date(ref.year, mes, dia)
    return (d if d >= ref else date(ref.year + 1, mes, dia)).isoformat()


def _ingles(t):
    cat = cargar_config()["catalogo"]["niveles_ingles"]
    inicio = t.find("ingles")
    zona = t[inicio:] if inicio >= 0 else t
    mejor = None
    for nivel, alias in cat.items():
        for a in _alias_norm(alias):
            m = re.search(rf"(?<![\w.]){re.escape(a)}(?![\w.])", zona)
            if m and (mejor is None or m.start() < mejor[0]):
                mejor = (m.start(), nivel, a)
    return _campo(mejor[1], mejor[2]) if mejor else _campo(None, None)


def extraer_entrevista(transcripcion: str) -> dict:
    t = normalizar_para_verificar(transcripcion)
    return {
        "anios_experiencia": _anios(t),
        "ultimo_puesto": _puesto(t),
        "tecnologias": _tecnologias(t),
        "pretension_salarial": _salario(t),
        "disponibilidad": _disponibilidad(t),
        "nivel_ingles": _ingles(t),
    }


# ---------------------------------------------------------------- Llamada

PREVIOS_CANTIDAD = {"sola", "solo", "porcion", "porciones", "jarra", "de", "sean"}
QUITAR = {"quito", "quita", "cambialo", "cambia", "sin"}


def _mascara_promos(tokens, promos):
    """Marca las posiciones que pertenecen a frases de promoción (no son productos)."""
    ocupado = [False] * len(tokens)
    for p in promos.values():
        for a in _alias_norm(p["alias"]):
            at = a.split()
            for i in range(len(tokens) - len(at) + 1):
                if tokens[i:i + len(at)] == at:
                    for k in range(i, i + len(at)):
                        ocupado[k] = True
    return ocupado


def _items(t):
    cat = cargar_config()["catalogo"]
    tokens = t.split()
    ocupado = _mascara_promos(tokens, cat["promociones"])
    alias = sorted(((pid, a.split()) for pid, p in cat["carta"].items() for a in _alias_norm(p["alias"])),
                   key=lambda x: len(x[1]), reverse=True)
    vistos: dict[str, dict] = {}
    for i in range(len(tokens)):
        for pid, at in alias:
            n = len(at)
            if tokens[i:i + n] != at or any(ocupado[i:i + n]):
                continue
            for k in range(i, i + n):
                ocupado[k] = True
            # cantidad: número inmediatamente antes, saltando "sola", "porcion de", "jarra de"...
            j, cantidad = i - 1, None
            while j >= 0 and i - j <= 4 and tokens[j] in PREVIOS_CANTIDAD:
                j -= 1
            if j >= 0 and tokens[j].isdigit() and int(tokens[j]) < 20:
                cantidad = int(tokens[j])
            # precio: "<producto> (esta|cuesta)? N soles" en las 5 palabras siguientes
            siguiente = " ".join(tokens[i + n: i + n + 5])
            mp = re.match(r"(?:\w+ ){0,2}?(\d+) soles", siguiente)
            precio = int(mp.group(1)) if mp else None
            quitado = any(w in QUITAR for w in tokens[max(0, i - 3): i])
            v = vistos.setdefault(pid, {"cantidad": None, "precio": None, "quitado": False, "ev": []})
            if cantidad is not None:
                v["cantidad"] = cantidad
            if precio is not None:
                v["precio"] = precio
            v["quitado"] = quitado
            v["ev"].append(" ".join(tokens[max(0, i - 2): i + n + 3]))
            break
    items, evid = [], []
    for pid, v in vistos.items():
        if v["quitado"] or (v["cantidad"] is None and v["precio"] is None):
            continue
        p = cat["carta"][pid]
        items.append({"producto": p["nombre"], "tamano": p["tamano"], "cantidad": v["cantidad"] or 1,
                      "precio_unitario": v["precio"]})
        evid.append(v["ev"][-1])
    return _campo(items or None, " | ".join(evid) or None)


def _promocion(t):
    cat = cargar_config()["catalogo"]
    negaciones = _alias_norm(cat["negaciones_promocion"])
    valida = None
    for pid, p in cat["promociones"].items():
        for a in _alias_norm(p["alias"]):
            for m in re.finditer(rf"(?<![\w.]){re.escape(a)}(?![\w.])", t):
                despues = " ".join(t[m.end():].split()[:14])
                if any(contiene(despues, n) for n in negaciones):
                    continue
                if valida is None or m.start() > valida[0]:
                    valida = (m.start(), {"nombre": p["nombre"], "descuento": p["descuento"]}, a)
    return _campo(valida[1], valida[2]) if valida else _campo(None, None)


def _direccion(texto_crudo):
    m = re.search(r"\b((?:jir[oó]n|avenida|calle|pasaje|av\.)\s[^.?!]+)", texto_crudo, flags=re.I)
    if not m:
        return _campo(None, None)
    valor = re.sub(r",?\s+en\s+(?=[A-ZÁÉÍÓÚ])", ", ", m.group(1).strip())
    return _campo(valor, normalizar_texto(m.group(1)))


def _pago(t):
    cat = cargar_config()["catalogo"]["metodos_pago"]
    inicio = t.find("pag")
    zona = t[inicio:] if inicio >= 0 else t
    mejor = None
    for metodo, alias in cat.items():
        for a in _alias_norm(alias):
            m = re.search(rf"(?<![\w.]){re.escape(a)}(?![\w.])", zona)
            if m and (mejor is None or m.start() < mejor[0]):
                mejor = (m.start(), metodo, a)
    return _campo(mejor[1], mejor[2]) if mejor else _campo(None, None)


def _total(t):
    ms = list(re.finditer(r"\btotal (?:es|seria|sera|son|queda en) (?:de )?(\d+)\b", t))
    return _campo(int(ms[-1].group(1)), ms[-1].group(0)) if ms else _campo(None, None)


def extraer_llamada(transcripcion: str) -> dict:
    t = normalizar_para_verificar(transcripcion)
    return {
        "items": _items(t),
        "promocion": _promocion(t),
        "direccion_entrega": _direccion(transcripcion),
        "metodo_pago": _pago(t),
        "total_dicho": _total(t),
    }
