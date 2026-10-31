"""Corre UNA vez: transcribe el corpus y guarda la caché en data/transcripciones/.

  python -m scripts.transcribir_corpus                 # L0 de los 10 clips, sin glosario
  python -m scripts.transcribir_corpus --glosario      # además hotwords e initial_prompt (entrevistas)
  python -m scripts.transcribir_corpus --escalera      # L1-L3 de las llamadas

Lo que ya está en caché no se vuelve a transcribir.
"""
import argparse

import soundfile as sf

from src.asr import transcribir
from src.config import cargar_config, ruta
from src.corpus import clips, guardar_cache, leer_cache, ruta_audio, variante_base
from src.degradacion import degradar


def _correr(clip, variante, audio, **kw):
    if leer_cache(clip, variante):
        return
    r = transcribir(audio, **kw)
    guardar_cache(clip, variante, r)
    print(f"{variante:<40} {clip}  {r['segundos_transcripcion']:>6.1f}s  {r['dispositivo_real']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosario", action="store_true")
    ap.add_argument("--escalera", action="store_true")
    a = ap.parse_args()
    base = variante_base()
    todos = clips("entrevista") + clips("llamada")

    for c in todos:
        _correr(c, f"{base}/L0", str(ruta_audio(c)), modo_glosario="ninguno")

    if a.glosario:
        for modo in ("hotwords", "initial_prompt"):
            for c in clips("entrevista"):
                _correr(c, f"{base}/glosario_{modo}", str(ruta_audio(c)), modo_glosario=modo)

    if a.escalera:
        fuentes = [str(ruta_audio(c)) for c in clips("entrevista")]
        carpeta = ruta("audio_degradado")
        for nivel in cargar_config()["degradacion"]["niveles"][1:]:
            for c in clips("llamada"):
                señal, meta = degradar(str(ruta_audio(c)), nivel, fuentes)
                (carpeta / nivel).mkdir(parents=True, exist_ok=True)
                sf.write(carpeta / nivel / f"{c}.wav", señal, 16000)
                if leer_cache(c, f"{base}/{nivel}"):
                    continue
                r = transcribir(señal, modo_glosario="ninguno")
                r["degradacion"] = {"nivel": nivel, **meta}
                guardar_cache(c, f"{base}/{nivel}", r)
                print(f"{base}/{nivel:<20} {c}  {meta}", flush=True)


if __name__ == "__main__":
    main()
