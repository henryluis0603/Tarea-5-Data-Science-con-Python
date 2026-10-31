"""Genera results/tablas.md (tablas en Markdown para el README) desde los JSON de resultados.

    python -m scripts.tablas_readme
"""
import json

import pandas as pd

from src.config import ruta
from src.corpus import ground_truth
from src.metricas import fecha_disponible


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    r = json.loads((ruta("resultados") / "resultados.json").read_text(encoding="utf-8"))
    partes = []

    ge = ground_truth("entrevista")
    dist = [{"clip": c, "anios_experiencia": g["anios_experiencia"], "ultimo_puesto": g["ultimo_puesto"],
             "n_tecnologias": len(g["tecnologias"]),
             "pretension": f"{g['pretension_salarial']['monto']} {g['pretension_salarial']['moneda']}",
             "disponibilidad": fecha_disponible(g["disponibilidad"]) + f" ({g['disponibilidad']['tipo']})",
             "nivel_ingles": g["nivel_ingles"]} for c, g in ge.items()]
    partes += ["### Corpus de entrevistas: distribución de los campos", md(pd.DataFrame(dist))]
    gl_ = ground_truth("llamada")
    dist = [{"clip": c, "n_items": len(g["items"]), "unidades": sum(i["cantidad"] for i in g["items"]),
             "promocion": (g["promocion"] or {}).get("nombre"), "metodo_pago": g["metodo_pago"],
             "total_dicho": g["total_dicho"], "total_correcto": g["total_correcto"],
             "autocorreccion": g["trampas"]["autocorreccion"], "total_erroneo": g["trampas"]["total_erroneo"],
             "tipo_error": g["trampas"]["tipo_error_total"]} for c, g in gl_.items()]
    partes += ["### Corpus de llamadas: distribución y trampas", md(pd.DataFrame(dist))]

    gpu_p = ruta("benchmark_gpu")
    if gpu_p.exists():
        g = json.loads(gpu_p.read_text(encoding="utf-8"))
        df = pd.DataFrame(g["filas"]).drop(columns=["texto"])
        t = (df.groupby(["modelo", "dispositivo_pedido", "dispositivo_real", "precision", "modo_glosario"])
             [["wer", "cer", "seg_por_min_audio", "vram_pico_mb", "vram_modelo_mb"]].mean().reset_index())
        partes += [f"### Benchmark GPU ({g['gpu']})", md(t.round(3))]
        filas = [{"modelo": m, "clip": x["clip"], "texto_identico": x["texto_identico"], "wer_entre_float16_e_int8": x["wer_entre_si"]}
                 for m, xs in g["precision_cambia_texto"].items() for x in xs]
        partes += ["### ¿Cambia el texto entre float16 e int8?", md(pd.DataFrame(filas))]
        partes += ["### ¿Cambia el texto entre GPU y CPU (int8)?", md(pd.DataFrame(g["cpu_vs_gpu_cambia_texto"]))]

    t1 = r["tarea1"]
    partes += ["### Tarea 1 — capa transcripción (caché local)", md(pd.DataFrame(t1["transcripcion"]))]
    c = pd.DataFrame(t1["campos"])
    c["exactitud (IC95)"] = c.apply(lambda x: f"{x.exactitud:.2f} [{x.ic95_inf:.2f}, {x.ic95_sup:.2f}]", axis=1)
    partes += ["### Tarea 1 — capa registro: exactitud por campo (n = 5, IC de Wilson)",
               md(c.pivot(index="campo", columns="metodo", values="exactitud (IC95)").reset_index())]
    partes += ["F1 macro de `nivel_ingles`: " + ", ".join(f"{k} = {v:.3f}" for k, v in t1["f1_macro_nivel_ingles"].items())]
    partes += ["### Tarea 1 — error absoluto medio", md(pd.DataFrame(t1["mae"]))]
    vt = pd.DataFrame(t1["verificacion_vs_correccion"])
    if len(vt):
        partes += ["### Verificación literal contra corrección", md(vt.pivot_table(index=["metodo", "verificacion"], columns="correcto", values="n", fill_value=0).reset_index())]

    gl = pd.DataFrame(r["glosario"]["comparaciones"])
    if len(gl):
        gl["ic95_dif"] = gl["ic95_dif"].apply(lambda x: f"[{x[0]:+.3f}, {x[1]:+.3f}]")
        partes += ["### Glosario (caché local, int8)", md(gl[["modo", "wer_sin", "wer_con", "dif_wer_media", "ic95_dif",
                                                              "terminos_total", "recuperados_sin", "recuperados_con"]])]

    t2 = r["tarea2"]
    esc = pd.DataFrame(t2["escalera"])
    partes += ["### Tarea 2 — escalera de degradación (ambas capas)",
               md(esc[["nivel", "metodo", "wer_medio", "cer_medio", "exactitud_campos", "exactitud_items",
                       "exactitud_total_dicho"]])]
    partes += ["### Tarea 2 — validación aritmética: detectados vs omitidos",
               md(esc[["nivel", "metodo", "trampas", "detectados", "omitidos", "falsas_alarmas",
                       "no_verificables", "diagnosticos_evaluados", "diagnosticos_correctos"]])]
    if t2["snr"]:
        partes += ["### SNR medida en L3", md(pd.DataFrame(t2["snr"]).query("nivel == 'L3'").dropna(axis=1, how="all"))]

    pts = pd.DataFrame(r["wer_vs_registro"])
    partes += ["### WER contra exactitud del registro (todas las combinaciones)",
               md(pts.sort_values("wer", ascending=False).head(8)), md(pts.sort_values("wer").head(8))]
    (ruta("resultados") / "tablas.md").write_text("\n\n".join(partes) + "\n", encoding="utf-8")
    print("Escrito results/tablas.md")


if __name__ == "__main__":
    main()
