"""Motor único: audio -> transcripción -> registro -> verificación (-> validación).

Una sola función pública, `procesar_audio`, sirve a las dos tareas: el esquema
("entrevista" o "llamada") elige el prompt, el extractor de reglas y si se valida la
aritmética. Este módulo no importa ninguna librería de interfaz.
"""
import time
import traceback

from .config import cargar_config
from .extraccion import llm, reglas
from .validacion import confianza_del_total, validar_pedido
from .verificacion import verificar

ESQUEMAS = {
    "entrevista": {
        "campos": ["anios_experiencia", "ultimo_puesto", "tecnologias", "pretension_salarial",
                   "disponibilidad", "nivel_ingles"],
        "prompt": "prompt_entrevista",
        "reglas": reglas.extraer_entrevista,
        "validar": False,
    },
    "llamada": {
        "campos": ["items", "promocion", "direccion_entrega", "metodo_pago", "total_dicho"],
        "prompt": "prompt_llamada",
        "reglas": reglas.extraer_llamada,
        "validar": True,
    },
}


def _canonizar(campos: dict) -> dict:
    """Formatos equivalentes -> forma canónica (solo formato, nunca el valor): 'S/.' -> 'PEN'."""
    sal = campos.get("pretension_salarial")
    if isinstance(sal, dict) and sal.get("moneda"):
        from .normalizacion import normalizar_texto
        m = str(sal["moneda"]).strip().lower()
        for iso, alias in cargar_config()["normalizacion"]["monedas"].items():
            if m in alias or normalizar_texto(m) in alias:
                sal["moneda"] = iso
    return campos


def procesar_audio(ruta_audio, esquema: str = "entrevista", metodo: str | None = None,
                   proveedor: str | None = None, transcripcion: dict | None = None) -> dict:
    """Procesa un audio y devuelve el resultado estructurado.

    ruta_audio    ruta del archivo (o array a 16 kHz).
    esquema       "entrevista" | "llamada".
    metodo        "llm" | "reglas" (por defecto, config.yaml).
    proveedor     "openai" | "ollama" para el método llm.
    transcripcion resultado previo de ASR (caché); si se pasa, no se vuelve a transcribir.
    """
    cfg = cargar_config()["extraccion"]
    metodo = metodo or cfg["metodo_defecto"]
    proveedor = proveedor or cfg["proveedor_defecto"]
    e = ESQUEMAS[esquema]
    salida = {
        "audio": str(ruta_audio) if isinstance(ruta_audio, str) else "<array>",
        "esquema": esquema, "metodo": metodo,
        "proveedor": proveedor if metodo == "llm" else None,
        "modelo_extraccion": llm.nombre_modelo(proveedor) if metodo == "llm" else "regex",
        "transcripcion": None, "campos": {}, "evidencias": {}, "verificacion": {},
        "validacion": None, "dispositivo": None, "modelo_asr": None,
        "segundos": {"asr": 0.0, "extraccion": 0.0}, "desde_cache": transcripcion is not None, "error": None,
    }
    try:
        if transcripcion is None:
            from .asr import transcribir   # import perezoso: evaluar desde caché no necesita Whisper
            transcripcion = transcribir(ruta_audio)
            salida["segundos"]["asr"] = transcripcion["segundos_transcripcion"]
        salida["transcripcion"] = transcripcion["texto"]
        salida["dispositivo"] = transcripcion.get("dispositivo_real")
        salida["modelo_asr"] = f"{transcripcion.get('modelo')} ({transcripcion.get('precision')})"

        t0 = time.perf_counter()
        if metodo == "reglas":
            crudo = e["reglas"](transcripcion["texto"])
        else:
            crudo = llm.extraer(transcripcion["texto"], e["prompt"], e["campos"], proveedor)
        salida["segundos"]["extraccion"] = round(time.perf_counter() - t0, 2)

        salida["campos"] = _canonizar({c: (crudo.get(c) or {}).get("valor") for c in e["campos"]})
        salida["evidencias"] = {c: (crudo.get(c) or {}).get("evidencia") for c in e["campos"]}
        salida["verificacion"] = verificar(transcripcion["texto"], salida["campos"])
        if e["validar"]:
            salida["validacion"] = validar_pedido(salida["campos"],
                                                  confianza_del_total(transcripcion.get("segmentos", [])))
    except Exception as exc:   # el error viaja en el resultado; la app decide cómo mostrarlo
        salida["error"] = f"{type(exc).__name__}: {exc}"
        salida["traza"] = traceback.format_exc(limit=3)
    salida["segundos"]["total"] = round(salida["segundos"]["asr"] + salida["segundos"]["extraccion"], 2)
    return salida
