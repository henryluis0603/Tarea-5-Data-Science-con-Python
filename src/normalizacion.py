"""Normalización de texto. La misma regla se aplica a referencia e hipótesis antes de compararlas.

Regla para WER/CER (`normalizar_texto`):
 1. minúsculas;
 2. se quitan tildes (la ñ se conserva);
 3. letras y dígitos pegados se separan ("BARRIO15" -> "barrio 15");
 4. se elimina toda puntuación;
 5. números en palabras -> dígitos (ver numeros.py);
 6. se eliminan las muletillas de config.yaml ("eh", "mmm").

Regla adicional para verificar valores (`normalizar_para_verificar`), específica del audio:
 7. "N lucas" -> "N000 soles" (jerga peruana);
 8. "N <unidad> y medio" -> "N.5 <unidad>" ("4 años y medio" -> "4.5 años");
 9. "una decada" -> "10 años";
10. "primero de <mes>" -> "1 de <mes>".
"""
import re
import unicodedata

from .config import cargar_config
from .numeros import palabras_a_digitos

MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"


def quitar_tildes(t: str) -> str:
    t = t.replace("ñ", "\0").replace("Ñ", "\0")
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return t.replace("\0", "ñ")


def normalizar_texto(texto: str) -> str:
    cfg = cargar_config()["normalizacion"]
    t = quitar_tildes(texto.lower())
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)
    t = re.sub(r"(?<=[a-zñ])(?=\d)|(?<=\d)(?=[a-zñ])", " ", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", "xpuntox", t)   # protege decimales ("4.5")
    t = re.sub(r"[^a-zñ0-9 ]+", " ", t)
    t = palabras_a_digitos(" ".join(t.split())).replace("xpuntox", ".")
    muletillas = set(cfg["muletillas"])
    return " ".join(w for w in t.split() if w not in muletillas)


def normalizar_para_verificar(texto: str) -> str:
    cfg = cargar_config()["normalizacion"]
    t = normalizar_texto(texto)
    jerga = "|".join(cfg["jerga_miles"])
    t = re.sub(rf"\b(\d+) (?:{jerga})\b", lambda m: f"{int(m.group(1)) * 1000} soles", t)
    t = re.sub(r"\b(\d+) (\w+) y medio\b", r"\1.5 \2", t)
    t = re.sub(r"\b(\d+) y medio\b", r"\1.5", t)
    t = re.sub(r"\b1 decada\b", "10 años", t)
    t = re.sub(rf"\bprimero de ({MESES})\b", r"1 de \1", t)
    return t


def contiene(texto_normalizado: str, fragmento: str) -> bool:
    """¿Aparece el fragmento (ya normalizado) como secuencia de palabras completa?"""
    if not fragmento:
        return False
    return re.search(rf"(?<![\w.]){re.escape(fragmento)}(?![\w.])", texto_normalizado) is not None
