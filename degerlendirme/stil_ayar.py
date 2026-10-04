# -*- coding: utf-8 -*-
"""
STİL AYARI — "duyarlilik" ve "baskin" değerlerini Seslen havuzlarının YALNIZ İLK yarısında (geliştirme)
seçer ve model/onnx/esikler.json'a "stil" olarak yazar. Ölçüm ikinci yarıda: degerlendirme/stil_havuz.py.
Amaç: ton başına doğruluğun ortalaması (sınıf dengeli); nötr örnek derlemin ayrı bir dilimi.

    python degerlendirme/stil_ayar.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from turkce_duygu import Cozumleyici  # noqa: E402
from turkce_duygu.eylem import eylem  # noqa: E402
from turkce_duygu.stil import stil_sec  # noqa: E402

HAVUZ = Path(os.environ.get("TURKCE_DUYGU_HAVUZ", Path.home() / "Desktop/gis-platform/sesatolye/veri/duygu"))
STILLER = ["neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin"]


def main() -> int:
    c = Cozumleyici(model=True)
    ornek: list[tuple[str, str]] = []
    for st in STILLER:
        ss = [s for s in (HAVUZ / f"{st}.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
        ornek += [(s, st) for s in ss[:len(ss) // 2]]
    cum = [json.loads(s)["metin"] for s in (HAVUZ.parent / "korpus.jsonl").read_text(encoding="utf-8").splitlines()]
    ornek += [(s, "notr") for s in [x for x in cum if x.endswith(".")][0::40][:400]]   # sınav dilimi 1::40
    metin = [s for s, _ in ornek]
    gucler, _ = c._gucler(metin)
    eylemler = [eylem(s) for s in metin]
    en_iyi = None
    for duy, bas in itertools.product([0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], [1.0, 1.25, 1.5, 2.0, 3.0, 99.0]):
        ayar = {"duyarlilik": duy, "baskin": bas}
        tahmin = [stil_sec(s, e, g, ayar=ayar).stil for s, e, g in zip(metin, eylemler, gucler)]
        oran = [np.mean([t == h for t, (_, h) in zip(tahmin, ornek) if h == st]) for st in STILLER + ["notr"]]
        puan = float(np.mean(oran))
        if en_iyi is None or puan > en_iyi[0]:
            en_iyi = (puan, ayar, oran)
    puan, ayar, oran = en_iyi
    print(f"seçilen {ayar}: geliştirme ton ortalaması %{100 * puan:.1f}")
    print("  " + "  ".join(f"{st} %{100 * o:.0f}" for st, o in zip(STILLER + ["notr"], oran)))
    yol = KOK / "model" / "onnx" / "esikler.json"
    d = json.loads(yol.read_text(encoding="utf-8"))
    d["stil"] = {**ayar, "gelistirme_ton_ortalamasi": round(puan, 4)}
    yol.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazıldı: {yol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
