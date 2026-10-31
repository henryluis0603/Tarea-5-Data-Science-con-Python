"""Gate de regresión para CI: recalcula métricas desde la caché (sin transcribir ni llamar a
APIs) y falla si la calidad cae bajo los umbrales de config.yaml.

Los runners de GitHub Actions no tienen GPU: transcribir ahí tardaría horas o fallaría.
Por eso el gate evalúa transcripciones y registros LLM ya guardados; el camino de reglas sí
se recalcula, así que cualquier cambio en reglas, normalización o verificación se mide.

    python -m scripts.regresion
"""
import sys

import numpy as np

from scripts.evaluar import METODOS, tarea1, tarea2
from src.config import cargar_config


def main() -> int:
    u = cargar_config()["regresion"]
    metodos = [m for m in METODOS if m[1] != "ollama"]
    t1, t2 = tarea1(metodos, solo_cache=True), tarea2(metodos, solo_cache=True)
    wer = float(np.mean([f["wer"] for f in t1["transcripcion"]]))
    ex_t1 = {f["metodo"]: f["exactitud"] for f in t1["campos"] if f["campo"] == "registro_completo"}
    camp_t1 = [f for f in t1["por_clip"] if f["metodo"] == "llm_openai"]
    ex_campos_t1 = float(np.mean([f["exactitud_registro"] for f in camp_t1])) if camp_t1 else 0.0
    camp_t2 = [f for f in t2["por_clip"] if f["metodo"] == "llm_openai" and f["nivel"] == "L0"]
    ex_campos_t2 = float(np.mean([f["exactitud_registro"] for f in camp_t2])) if camp_t2 else 0.0

    checks = [
        ("WER medio entrevistas", wer, "<=", u["wer_max_entrevistas"]),
        ("Exactitud por campo entrevistas (LLM)", ex_campos_t1, ">=", u["exactitud_min_entrevistas_llm"]),
        ("Exactitud por campo llamadas L0 (LLM)", ex_campos_t2, ">=", u["exactitud_min_llamadas_llm_L0"]),
    ]
    fallo = False
    for nombre, valor, op, umbral in checks:
        ok = valor <= umbral if op == "<=" else valor >= umbral
        fallo |= not ok
        print(f"{'OK  ' if ok else 'FAIL'} {nombre}: {valor:.3f} {op} {umbral}")
    print("Registro completo T1:", {k: round(v, 2) for k, v in ex_t1.items()})
    return 1 if fallo else 0


if __name__ == "__main__":
    sys.exit(main())
