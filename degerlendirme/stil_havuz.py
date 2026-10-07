# -*- coding: utf-8 -*-
"""
STİL SINAMASI — tam çözümleyicinin (kurallar + model) seçtiği seslendirme stili, Seslen ses atölyesinin
duygu havuzlarında. Havuzlar yayımlanmıyor; elindeyse çalışır (TURKCE_DUYGU_HAVUZ).
Yalnız İKİNCİ yarılar (kurallar geliştirilirken hiç incelenmedi). Fısıltı metinden seçilmez, ölçülmez.
Nötr örnek: ses atölyesi derleminin "." ile biten cümleleri (veri/korpus.jsonl).

    python degerlendirme/stil_havuz.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turkce_duygu import Cozumleyici  # noqa: E402

HAVUZ = Path(os.environ.get("TURKCE_DUYGU_HAVUZ", Path.home() / "Desktop/gis-platform/seslen/veri/duygu"))
STILLER = ["neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin"]


def main() -> int:
    c = Cozumleyici(model=True)
    toplam_d = toplam_n = 0
    for st in STILLER:
        ss = [s for s in (HAVUZ / f"{st}.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
        sinav = ss[len(ss) // 2:]
        sonuc = [c.metin_cozumle(s, bol=False)[0].stil for s in sinav]
        d = sum(x == st for x in sonuc)
        toplam_d, toplam_n = toplam_d + d, toplam_n + len(sinav)
        print(f"{st:8s} {d:4d}/{len(sinav)}  %{100 * d / len(sinav):.1f}   {Counter(sonuc).most_common(4)}")
    korpus = HAVUZ.parent / "korpus.jsonl"
    if korpus.exists():
        cum = [json.loads(s)["metin"] for s in korpus.read_text(encoding="utf-8").splitlines()]
        cum = [x for x in cum if x.endswith(".")][1::40][:400]
        sonuc = [c.metin_cozumle(s, bol=False)[0].stil for s in cum]
        d = sum(x == "notr" for x in sonuc)
        print(f"{'notr':8s} {d:4d}/{len(cum)}  %{100 * d / len(cum):.1f}   {Counter(sonuc).most_common(4)}")
        toplam_d, toplam_n = toplam_d + d, toplam_n + len(cum)
    print(f"TOPLAM {toplam_d}/{toplam_n}  %{100 * toplam_d / toplam_n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
