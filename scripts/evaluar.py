"""Evalúa ambas tareas desde la caché de transcripciones (no transcribe nada).

  python -m scripts.evaluar                # reglas + OpenAI + Ollama
  python -m scripts.evaluar --sin-ollama   # solo reglas + OpenAI
  python -m scripts.evaluar --solo-cache   # no llama a ningún LLM: usa registros ya guardados

Los registros de cada método se guardan en results/registros/ para no volver a pagar la API.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import cargar_config, ruta
from src.corpus import clips, ground_truth, leer_cache, referencia, ruta_audio, variante_base
from src.engine import ESQUEMAS, procesar_audio
from src.metricas import (bootstrap_media, f1_macro, items_por_producto, puntuar_entrevista,
                          puntuar_llamada, wer_cer, wilson)
from src.normalizacion import contiene, normalizar_texto

METODOS = [("reglas", None), ("llm", "openai"), ("llm", "ollama")]


def nombre_metodo(m, p):
    return "reglas" if m == "reglas" else f"llm_{p}"


def registro(clip, esquema, nivel, metodo, proveedor, solo_cache):
    destino = ruta("resultados") / "registros" / variante_base() / nivel / nombre_metodo(metodo, proveedor) / f"{clip}.json"
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    if solo_cache and metodo == "llm":
        return None
    tr = leer_cache(clip, f"{variante_base()}/{nivel}")
    if tr is None:
        return None
    r = procesar_audio(str(ruta_audio(clip)), esquema, metodo, proveedor, transcripcion=tr)
    if r["error"]:
        print("  ERROR", clip, nivel, metodo, proveedor, r["error"])
        return r
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def resumen_campos(filas: pd.DataFrame, campos: list[str]) -> list[dict]:
    salida = []
    for (metodo,), g in filas.groupby(["metodo"]):
        for c in campos + ["registro_completo"]:
            k, n = int(g[c].sum()), len(g)
            lo, hi = wilson(k, n)
            salida.append({"metodo": metodo, "campo": c, "aciertos": k, "n": n,
                           "exactitud": k / n, "ic95_inf": lo, "ic95_sup": hi})
    return salida


# ------------------------------------------------------------------ Tarea 1

def tarea1(metodos, solo_cache):
    gt = ground_truth("entrevista")
    campos = ESQUEMAS["entrevista"]["campos"]
    base = variante_base()
    filas, transcripcion, verif = [], [], []
    for clip in clips("entrevista"):
        tr = leer_cache(clip, f"{base}/L0")
        m = wer_cer(referencia(clip), tr["texto"])
        transcripcion.append({"clip": clip, **m, "dispositivo": tr["dispositivo_real"],
                              "seg_por_min_audio": tr["seg_por_min_audio"]})
        for metodo, prov in metodos:
            r = registro(clip, "entrevista", "L0", metodo, prov, solo_cache)
            if not r or r.get("error"):
                continue
            p = puntuar_entrevista(r["campos"], gt[clip])
            fila = {"clip": clip, "metodo": nombre_metodo(metodo, prov), "wer": m["wer"],
                    **{c: p[c]["correcto"] for c in campos},
                    "error_abs_anios": p["anios_experiencia"]["error_abs"],
                    "error_abs_salario": p["pretension_salarial"]["error_abs"],
                    "f1_tecnologias": p["tecnologias"]["f1"],
                    "nivel_ingles_pred": r["campos"].get("nivel_ingles")}
            fila["registro_completo"] = all(fila[c] for c in campos)
            fila["exactitud_registro"] = float(np.mean([fila[c] for c in campos]))
            filas.append(fila)
            for c in campos:
                verif.append({"metodo": fila["metodo"], "clip": clip, "campo": c,
                              "verificacion": r["verificacion"][c], "correcto": p[c]["correcto"]})
    df = pd.DataFrame(filas)
    f1s = {met: f1_macro([gt[c]["nivel_ingles"] for c in g["clip"]], list(g["nivel_ingles_pred"]))
           for met, g in df.groupby("metodo")}
    mae = df.groupby("metodo")[["error_abs_anios", "error_abs_salario"]].mean().reset_index().to_dict("records")
    vdf = pd.DataFrame(verif)
    tabla_verif = (vdf.groupby(["metodo", "verificacion", "correcto"]).size().reset_index(name="n").to_dict("records"))
    return {"transcripcion": transcripcion, "por_clip": filas, "campos": resumen_campos(df, campos),
            "f1_macro_nivel_ingles": f1s, "mae": mae, "verificacion_vs_correccion": tabla_verif}


def glosario():
    """WER y recuperación de términos del glosario: sin glosario vs hotwords vs initial_prompt."""
    base = variante_base()
    terminos = [normalizar_texto(x) for x in cargar_config()["asr"]["glosario"].split(",")]
    filas = []
    for clip in clips("entrevista"):
        ref = normalizar_texto(referencia(clip))
        presentes = [t for t in terminos if t and contiene(ref, t)]
        for variante in ("L0", "glosario_hotwords", "glosario_initial_prompt"):
            tr = leer_cache(clip, f"{base}/{variante}")
            if tr is None:
                continue
            hip = normalizar_texto(tr["texto"])
            filas.append({"clip": clip, "modo": "ninguno" if variante == "L0" else variante.replace("glosario_", ""),
                          **wer_cer(referencia(clip), tr["texto"]),
                          "terminos_en_ref": len(presentes),
                          "terminos_recuperados": sum(contiene(hip, t) for t in presentes)})
    df = pd.DataFrame(filas)
    comparaciones = []
    if len(df):
        piv = df.pivot(index="clip", columns="modo", values="wer")
        rec = df.pivot(index="clip", columns="modo", values="terminos_recuperados")
        tot = df.pivot(index="clip", columns="modo", values="terminos_en_ref")["ninguno"]
        for modo in [m for m in ("hotwords", "initial_prompt") if m in piv]:
            dif = (piv[modo] - piv["ninguno"]).dropna()
            lo, hi = bootstrap_media(dif.values)
            rec_sin, rec_con = int(rec["ninguno"].sum()), int(rec[modo].sum())
            n = int(tot.sum())
            comparaciones.append({"modo": modo, "wer_sin": float(piv["ninguno"].mean()), "wer_con": float(piv[modo].mean()),
                                  "dif_wer_media": float(dif.mean()), "ic95_dif": [lo, hi],
                                  "terminos_total": n, "recuperados_sin": rec_sin, "recuperados_con": rec_con,
                                  "recall_sin": rec_sin / n, "recall_con": rec_con / n,
                                  "ic95_recall_sin": wilson(rec_sin, n), "ic95_recall_con": wilson(rec_con, n)})
    return {"por_clip": filas, "comparaciones": comparaciones}


# ------------------------------------------------------------------ Tarea 2

def causa_real(pred, g):
    """Causa verdadera de un descuadre: 'asr' si el registro extraído difiere del verdadero en
    ítems, descuento o total; 'hablante' si el registro es fiel y el total dicho estaba mal."""
    fiel = (items_por_producto(pred.get("items")) == items_por_producto(g["items"])
            and puntuar_llamada(pred, g)["promocion"]["correcto"]
            and puntuar_llamada(pred, g)["total_dicho"]["correcto"])
    if not fiel:
        return "asr"
    return "hablante" if g["trampas"]["total_erroneo"] else None


def tarea2(metodos, solo_cache):
    gt = ground_truth("llamada")
    campos = ESQUEMAS["llamada"]["campos"]
    base = variante_base()
    niveles = cargar_config()["degradacion"]["niveles"]
    filas, wer_niv, snr = [], [], []
    for nivel in niveles:
        for clip in clips("llamada"):
            tr = leer_cache(clip, f"{base}/{nivel}")
            if tr is None:
                continue
            m = wer_cer(referencia(clip), tr["texto"])
            wer_niv.append({"nivel": nivel, "clip": clip, **m})
            if tr.get("degradacion"):
                snr.append({"nivel": nivel, "clip": clip, **tr["degradacion"]})
            for metodo, prov in metodos:
                r = registro(clip, "llamada", nivel, metodo, prov, solo_cache)
                if not r or r.get("error"):
                    continue
                p = puntuar_llamada(r["campos"], gt[clip])
                v = r["validacion"] or {}
                roto = gt[clip]["trampas"]["total_erroneo"]
                marcado = v.get("estado") == "no_cuadra"
                cr = causa_real(r["campos"], gt[clip]) if marcado else None
                fila = {"nivel": nivel, "clip": clip, "metodo": nombre_metodo(metodo, prov), "wer": m["wer"],
                        **{c: p[c]["correcto"] for c in campos}, "f1_items": p["items"]["f1"],
                        "trampa_total_erroneo": roto, "estado_validacion": v.get("estado"),
                        "detectado": roto and marcado, "omitido": roto and not marcado,
                        "falsa_alarma": (not roto) and marcado,
                        "causa_diagnosticada": v.get("causa"), "causa_real": cr,
                        "diagnostico_correcto": (v.get("causa") == cr) if (marcado and cr) else None,
                        "explicacion": v.get("explicacion"), "calculado": v.get("calculado"),
                        "total_dicho_extraido": v.get("total_dicho")}
                fila["registro_completo"] = all(fila[c] for c in campos)
                fila["exactitud_registro"] = float(np.mean([fila[c] for c in campos]))
                filas.append(fila)
    df = pd.DataFrame(filas)
    escalera = []
    for (nivel, metodo), g in df.groupby(["nivel", "metodo"]):
        w = pd.DataFrame(wer_niv).query("nivel == @nivel")
        diag = g["diagnostico_correcto"].dropna()
        escalera.append({"nivel": nivel, "metodo": metodo, "wer_medio": float(w["wer"].mean()),
                         "cer_medio": float(w["cer"].mean()),
                         "exactitud_campos": float(g["exactitud_registro"].mean()),
                         **{f"exactitud_{c}": float(g[c].mean()) for c in campos},
                         "trampas": int(g["trampa_total_erroneo"].sum()), "detectados": int(g["detectado"].sum()),
                         "omitidos": int(g["omitido"].sum()), "falsas_alarmas": int(g["falsa_alarma"].sum()),
                         "no_verificables": int((g["estado_validacion"] == "no_verificable").sum()),
                         "diagnosticos_evaluados": int(len(diag)), "diagnosticos_correctos": int(diag.sum())})
    return {"por_clip": filas, "escalera": escalera, "wer_por_nivel": wer_niv, "snr": snr,
            "campos_L0": resumen_campos(df.query("nivel == 'L0'"), campos)}


def wer_vs_registro(t1, t2):
    """Todas las combinaciones (clip, nivel, método) con su WER y la exactitud de su registro."""
    puntos = [{"tarea": "entrevista", "nivel": "L0", **{k: f[k] for k in ("clip", "metodo", "wer", "exactitud_registro")}}
              for f in t1["por_clip"]]
    puntos += [{"tarea": "llamada", **{k: f[k] for k in ("clip", "nivel", "metodo", "wer", "exactitud_registro")}}
               for f in t2["por_clip"]]
    return puntos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-ollama", action="store_true")
    ap.add_argument("--solo-cache", action="store_true")
    a = ap.parse_args()
    metodos = [m for m in METODOS if not (a.sin_ollama and m[1] == "ollama")]
    salida = ruta("resultados")
    salida.mkdir(exist_ok=True)
    t1, t2 = tarea1(metodos, a.solo_cache), tarea2(metodos, a.solo_cache)
    datos = {"variante_asr": variante_base(), "tarea1": t1, "glosario": glosario(), "tarea2": t2,
             "wer_vs_registro": wer_vs_registro(t1, t2)}
    (salida / "resultados.json").write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(t1["campos"]).to_csv(salida / "t1_campos.csv", index=False)
    pd.DataFrame(t1["transcripcion"]).to_csv(salida / "t1_transcripcion.csv", index=False)
    pd.DataFrame(t2["escalera"]).to_csv(salida / "t2_escalera.csv", index=False)
    pd.DataFrame(t2["por_clip"]).to_csv(salida / "t2_por_clip.csv", index=False)
    print("Resultados en", salida)


if __name__ == "__main__":
    main()
