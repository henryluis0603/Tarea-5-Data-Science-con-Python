"""Números dichos en palabras -> dígitos ("seis mil quinientos" -> "6500").

Reglas (documentadas en el README):
- Se convierten secuencias completas: unidades, decenas, "veinti-", centenas, "mil", "millón".
- "y" solo une decena (30-90) con unidad (1-9): "treinta y cinco" -> 35.
- "un", "una", "uno" cuentan como 1 ("una jarra" -> "1 jarra"), igual en referencia e hipótesis.
- "medio" NO se convierte aquí, porque también es un tamaño ("medio pollo").
- Los separadores de miles que escribe el ASR ("6.500", "6,500") se eliminan.
"""
import re

UNIDADES = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
ESPECIALES = {  # 10-29: no admiten nada después salvo "mil"/"millón"
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiun": 21, "veintiuno": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}
DECENAS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90,
}
CENTENAS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
    "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500,
    "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
    "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900,
    "novecientas": 900,
}
MULTIPLICADORES = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}

PALABRAS_NUMERO = set(UNIDADES) | set(ESPECIALES) | set(DECENAS) | set(CENTENAS) | set(MULTIPLICADORES)


def _leer_numero(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Lee el número más largo que empieza en tokens[i]. Devuelve (valor, índice siguiente)."""
    total, actual, j = 0, 0, i
    limite = 999          # valor máximo que puede sumarse al grupo actual
    usado = False
    multiplicador_previo = 10**9
    while j < len(tokens):
        w = tokens[j]
        if w in MULTIPLICADORES:
            m = MULTIPLICADORES[w]
            if m >= multiplicador_previo or (not usado and w != "mil"):
                break
            total += (actual or 1) * m
            actual, limite, usado, multiplicador_previo = 0, 999, True, m
            j += 1
            continue
        if w == "y" and actual % 100 in DECENAS.values() and j + 1 < len(tokens) \
                and tokens[j + 1] in UNIDADES and UNIDADES[tokens[j + 1]] > 0:
            actual += UNIDADES[tokens[j + 1]]
            limite, j = 0, j + 2
            continue
        valor = None
        nuevo_limite = 0
        if w in CENTENAS:
            valor, nuevo_limite = CENTENAS[w], (0 if w == "cien" else 99)
        elif w in DECENAS:
            valor, nuevo_limite = DECENAS[w], 0   # la unidad solo entra con "y"
        elif w in ESPECIALES:
            valor = ESPECIALES[w]
        elif w in UNIDADES:
            valor = UNIDADES[w]
        if valor is None or valor > limite or (valor >= 100 and actual % 1000 != 0):
            break
        actual += valor
        limite, usado = nuevo_limite, True
        j += 1
    if not usado:
        return None
    return total + actual, j


def palabras_a_digitos(texto: str) -> str:
    """Reemplaza cada número escrito en palabras por sus dígitos. Espera texto en minúsculas sin tildes."""
    texto = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", texto)
    tokens = texto.split()
    salida, i = [], 0
    while i < len(tokens):
        leido = _leer_numero(tokens, i) if tokens[i] in PALABRAS_NUMERO else None
        if leido:
            valor, i = leido
            salida.append(str(valor))
        else:
            salida.append(tokens[i])
            i += 1
    return " ".join(salida)
