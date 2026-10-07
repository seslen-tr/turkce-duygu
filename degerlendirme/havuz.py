# -*- coding: utf-8 -*-
"""
SESLEN DUYGU HAVUZLARIYLA SINAMA — eylem kurallarının bağımsız bir denetimi.

Seslen ses atölyesinin eğitim için yazdırdığı duygu cümleleri (merak, uyarı, özür; 700'er cümle)
yayımlanmıyor; bu betik onlar elindeyse çalışır:  TURKCE_DUYGU_HAVUZ=<klasör> python degerlendirme/havuz.py
Kurallar YALNIZ ilk yarıya (geliştirme) bakılarak düzeltildi; ikinci yarı (sınav) hiç incelenmedi.
Örnek dökümü yalnız geliştirme yarısından yapılır (--goster).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turkce_duygu.eylem import eylem  # noqa: E402

HAVUZ = Path(os.environ.get("TURKCE_DUYGU_HAVUZ", Path.home() / "Desktop/gis-platform/seslen/veri/duygu"))
BEKLENEN = {"merak": {"soru", "danisma"}, "uyari": {"uyari", "yasak"}, "ozur": {"ozur"}}


def main() -> int:
    goster = "--goster" in sys.argv
    for ad, hedef in BEKLENEN.items():
        ss = [s for s in (HAVUZ / f"{ad}.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
        yari = len(ss) // 2
        for bolum, parca in (("geliştirme", ss[:yari]), ("sınav", ss[yari:])):
            turler = [eylem(s).tur for s in parca]
            d = sum(t in hedef for t in turler)
            print(f"{ad:6s} {bolum:10s} {d:4d}/{len(parca)}  %{100 * d / len(parca):.1f}  {Counter(turler).most_common(4)}")
        if goster:
            for s in [s for s in ss[:yari] if eylem(s).tur not in hedef][:15]:
                print("   ", eylem(s).tur, "|", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
