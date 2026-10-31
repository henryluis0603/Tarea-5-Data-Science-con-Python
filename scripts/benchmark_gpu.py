"""Benchmark de transcripción en GPU (correr en Google Colab con GPU T4).

Compara, sobre las entrevistas:
  - dos tamaños de modelo (config: benchmark.modelos);
  - GPU contra CPU con el mismo modelo;
  - dos precisiones en GPU (float16 contra int8), incluyendo si el TEXTO cambió;
  - glosario: sin glosario, hotwords e initial_prompt (modelo grande, float16).
Reporta WER, CER, segundos por minuto de audio, dispositivo leído en tiempo de ejecución y
VRAM pico medida con NVML. Guarda todo en results/benchmark_gpu.json y las transcripciones
en data/transcripciones/benchmark/.

  python -m scripts.benchmark_gpu
  python -m scripts.benchmark_gpu --clips-cpu E01 E02   # CPU es lenta: limita los clips
"""
import argparse
import gc
import json

import jiwer

from src import asr
from src.config import cargar_config, ruta
from src.corpus import guardar_cache, referencia, ruta_audio
from src.metricas import wer_cer
from src.normalizacion import normalizar_texto


def vram_usada_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**20
    except Exception:
        return None


def liberar():
    asr._MODELOS.clear()
    gc.collect()


def correr(modelo, dispositivo, precision, clips, modo_glosario="ninguno"):
    liberar()
    vram_antes = vram_usada_mb()
    filas = []
    for clip in clips:
        r = asr.transcribir(str(ruta_audio(clip)), modelo=modelo, dispositivo=dispositivo,
                            precision=precision, modo_glosario=modo_glosario)
        variante = f"benchmark/{modelo}_{dispositivo}_{precision}_{modo_glosario}"
        guardar_cache(clip, variante, r)
        m = wer_cer(referencia(clip), r["texto"])
        filas.append({"clip": clip, "modelo": modelo, "dispositivo_pedido": dispositivo,
                      "dispositivo_real": r["dispositivo_real"], "precision": precision,
                      "modo_glosario": modo_glosario, **m,
                      "seg_por_min_audio": r["seg_por_min_audio"], "segundos": r["segundos_transcripcion"],
                      "vram_antes_carga_mb": vram_antes, "vram_pico_mb": r["vram_pico_mb"],
                      "vram_modelo_mb": (r["vram_pico_mb"] - vram_antes) if (r["vram_pico_mb"] and vram_antes is not None) else None,
                      "texto": r["texto"], "faster_whisper": r["faster_whisper"], "ctranslate2": r["ctranslate2"]})
        print(f"{modelo:<16} {dispositivo:<5} {precision:<8} {modo_glosario:<15} {clip} "
              f"WER={m['wer']:.3f} {r['seg_por_min_audio']:.1f}s/min dev={r['dispositivo_real']} "
              f"VRAM={r['vram_pico_mb']}", flush=True)
    return filas


def cambios_texto(filas, a, b):
    """WER entre las transcripciones de dos configuraciones: 0 = texto idéntico."""
    salida = []
    for clip in {f["clip"] for f in filas}:
        ta = next((f["texto"] for f in filas if f["clip"] == clip and a(f)), None)
        tb = next((f["texto"] for f in filas if f["clip"] == clip and b(f)), None)
        if ta and tb:
            na, nb = normalizar_texto(ta), normalizar_texto(tb)
            salida.append({"clip": clip, "texto_identico": na == nb, "wer_entre_si": jiwer.wer(na, nb)})
    return sorted(salida, key=lambda x: x["clip"])


def main():
    cfg = cargar_config()["benchmark"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-cpu", nargs="*", default=cfg["clips"][:2])
    a = ap.parse_args()
    clips = cfg["clips"]
    if asr.gpus_disponibles() == 0:
        print("AVISO: no hay GPU visible para CTranslate2. Activa la GPU en Colab.")
    filas = []
    for modelo in cfg["modelos"]:
        for precision in cfg["precisiones_gpu"]:
            filas += correr(modelo, "cuda", precision, clips)
    grande = cfg["modelos"][-1]
    filas += correr(grande, "cpu", cfg["precision_cpu"], a.clips_cpu)
    for modo in ("hotwords", "initial_prompt"):
        filas += correr(grande, "cuda", "float16", clips, modo)

    p16, p8 = cfg["precisiones_gpu"]
    es = lambda m, d, p, g="ninguno": (lambda f: f["modelo"] == m and f["dispositivo_pedido"] == d
                                       and f["precision"] == p and f["modo_glosario"] == g)
    resultado = {
        "filas": filas,
        "precision_cambia_texto": {m: cambios_texto(filas, es(m, "cuda", p16), es(m, "cuda", p8)) for m in cfg["modelos"]},
        "cpu_vs_gpu_cambia_texto": cambios_texto(filas, es(grande, "cuda", cfg["precision_cpu"]), es(grande, "cpu", cfg["precision_cpu"])),
        "gpu": __import__("subprocess").run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
    }
    destino = ruta("benchmark_gpu")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Guardado en", destino)


if __name__ == "__main__":
    main()
