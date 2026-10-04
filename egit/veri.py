# -*- coding: utf-8 -*-
"""
VERİ — eğitim ve ölçüm verisini indirir, tekrarlanabilir biçimde böler.  python egit/veri.py

1) XED (Öhman vd. 2020, CC BY 4.0): film altyazılarında Plutchik'in 8 temel duygusu. Türkçe bölüm
   İngilizce/Fince insan etiketlerinin paralel altyazılara aktarımıdır (annotation projection);
   9.2 bin satır, çoğu tek etiketli. Sürüm sabit (commit aşağıda).
2) Nötr örnek: XED Türkçe aktarımında "nötr" yok; yalnız duygulu cümlelerle eğitilen model her cümleye
   bir duygu biçiyordu (03.10, ilk deneme). Nötr örnekler (etiketsiz satır olarak eğitime girer):
   Türkçe Vikipedi (CC BY-SA 4.0, wikimedia/wikipedia 20231101.tr) ve Common Voice Türkçe cümle derlemi
   (CC0, sabit sürüm) — duygu sözcüğü, tırnak, soru ve ünlem içermeyen düz bildirimler.

Bölme: cümlenin MD5 özetine göre %80 eğitim, %10 doğrulama, %10 sınav (aynı cümle hep aynı bölümde).
Çıktı veri/: xed_{egitim,dogrulama,sinav}.tsv (cümle<TAB>etiketler), notr_{egitim,dogrulama,sinav}.txt
"""
from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turkce_duygu.model import hazirla  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "veri"
XED_SURUM = "76666d251f147cf518071bef4f8415dab6b96115"
XED_URL = f"https://raw.githubusercontent.com/Helsinki-NLP/XED/{XED_SURUM}/Projections/tr-projections2.tsv"
# XED numaraları (1-8, alfabetik İngilizce adlar): anger, anticipation, disgust, fear, joy, sadness, surprise, trust
DUYGULAR = ("ofke", "bekleyis", "tiksinti", "korku", "sevinc", "uzuntu", "saskinlik", "guven")


def bolum(cumle: str) -> str:
    h = int(hashlib.md5(cumle.encode("utf-8")).hexdigest(), 16) % 10
    return "sinav" if h == 0 else "dogrulama" if h == 1 else "egitim"


def xed() -> dict[str, list[tuple[str, list[str]]]]:
    ham = urllib.request.urlopen(XED_URL, timeout=120).read().decode("utf-8")
    satirlar: dict[str, set[str]] = {}
    for p in list(csv.reader(io.StringIO(ham), delimiter="\t"))[1:]:
        if len(p) < 2 or not p[0].strip():
            continue
        c = hazirla(p[0])
        etiket = {DUYGULAR[int(x) - 1] for x in p[1].replace(" ", "").split(",") if x.isdigit() and 1 <= int(x) <= 8}
        if etiket:
            satirlar.setdefault(c, set()).update(etiket)       # aynı cümle iki kez geçerse etiketler birleşir
    sonuc: dict[str, list] = {"egitim": [], "dogrulama": [], "sinav": []}
    for c, e in satirlar.items():
        sonuc[bolum(c)].append((c, sorted(e, key=DUYGULAR.index)))
    return sonuc


def vikipedi_notr(adet: int = 3000) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    yol = hf_hub_download("wikimedia/wikipedia", "20231101.tr/train-00000-of-00002.parquet", repo_type="dataset")
    metin = pq.read_table(yol, columns=["text"]).column("text").to_pylist()
    rnd = random.Random(20231101)
    rnd.shuffle(metin)
    cumleler: list[str] = []
    for m in metin:
        for paragraf in m.split("\n"):
            for c in re.split(r"(?<=[.!?])\s+", paragraf.strip()):
                n = len(c.split())
                # tam cümle, 5-25 sözcük, başlık/liste/kaynakça değil, sayı yığını değil
                if 5 <= n <= 25 and c[:1].isupper() and c.endswith(".") and sum(ch.isdigit() for ch in c) < 6 \
                        and not re.search(r"[|=<>{}\[\]]", c):
                    cumleler.append(c)
                    break                          # makale başına en çok bir cümle: çeşitlilik
            if len(cumleler) >= adet:
                return cumleler
            break
    return cumleler


CV_SURUM = "b12cde653757295103c5948c1cd601b166e1b905"
CV_URL = f"https://raw.githubusercontent.com/common-voice/common-voice/{CV_SURUM}/server/data/tr/sentence-collector.txt"
# Duygu bildiren sözcük kökleri: bu cümleler "nötr" örneğe alınmaz (etiketsiz duygulu cümle nötr diye öğretilmesin).
_DUYGU_SOZ = re.compile(r"(?i)\b(?:üz[gü]|mutlu|sevin|kork|kız[gd]|öfke|ağla|gül[eü]|neşe|heyecan|şaşır|hayret|"
                        r"nefret|tiksin|iğren|endişe|kayg|umut|özle|sev[ei]|aşk|acı|keder|hüzün|yazık|eyvah|harika|"
                        r"muhteşem|berbat|rezil|lanet|kahr|bıkt|merak|utan|pişman|gurur|yalnız|ölü|öld)")


def common_voice_notr(adet: int = 4000) -> list[str]:
    """Common Voice Türkçe cümle derlemi (CC0; çoğu edebî alıntı ve günlük dil). Duygusuz düz bildirimler:
    tırnak / soru / ünlem / üç nokta yok, duygu sözcüğü yok, 5-20 sözcük."""
    ham = urllib.request.urlopen(CV_URL, timeout=120).read().decode("utf-8").splitlines()
    uygun = [c.strip() for c in ham if 5 <= len(c.split()) <= 20 and c.strip().endswith(".") and c[:1].isupper()
             and not re.search(r"[\"“”'‘’!?…:;()]|\.\.", c.strip()[:-1]) and not _DUYGU_SOZ.search(c)]
    random.Random(48).shuffle(uygun)
    return uygun[:adet]


def main() -> int:
    VERI.mkdir(exist_ok=True)
    x = xed()
    for ad, satir in x.items():
        with open(VERI / f"xed_{ad}.tsv", "w", encoding="utf-8", newline="\n") as f:
            for c, e in satir:
                f.write(f"{c}\t{','.join(e)}\n")
        print(f"xed_{ad}.tsv  {len(satir)}")
    # Nötr: Vikipedi (ansiklopedik) + Common Voice (günlük / edebî dil); aynı MD5 bölmesi
    notr = {"egitim": [], "dogrulama": [], "sinav": []}
    for c in vikipedi_notr(4000) + common_voice_notr(4000):
        notr[bolum(c)].append(hazirla(c))
    for ad, satir in notr.items():
        (VERI / f"notr_{ad}.txt").write_text("\n".join(satir) + "\n", encoding="utf-8")
        print(f"notr_{ad}.txt {len(satir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
