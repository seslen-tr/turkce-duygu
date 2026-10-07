# -*- coding: utf-8 -*-
"""
SIZINTI TEMİZLİĞİ — açık stil setinden (turkce_duygu/veri/stil_seti) sınavda kullanılan Seslen
havuzlarındaki cümlelere çok benzeyenleri çıkarır (karakter 3-gram Jaccard ≥ 0,5). Böylece model sınav
cümlelerini ya da yakın kopyalarını eğitimde görmez. Set içi yinelenenler de atılır.

    python degerlendirme/sizinti_temizle.py
"""
from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
SET = KOK / "turkce_duygu" / "veri" / "stil_seti"
HAVUZ = Path(os.environ.get("TURKCE_DUYGU_HAVUZ", Path.home() / "Desktop/gis-platform/seslen/veri/duygu"))
ESIK = 0.5


def gramlar(s: str) -> set[str]:
    k = " " + unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i")).lower() + " "
    return {k[i:i + 3] for i in range(len(k) - 2)}


def main() -> int:
    havuz = [gramlar(s) for f in HAVUZ.glob("*.txt") for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
    dizin: dict[str, list[int]] = {}
    for i, g in enumerate(havuz):
        for x in g:
            dizin.setdefault(x, []).append(i)
    toplam_at = 0
    for f in sorted(SET.glob("*.txt")):
        satir = [s.strip() for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
        kalan, gorulen, atilan = [], set(), []
        for s in satir:
            g = gramlar(s)
            aday: dict[int, int] = {}
            for x in g:
                for i in dizin.get(x, ()):
                    aday[i] = aday.get(i, 0) + 1
            benzer = max((n / len(g | havuz[i]) for i, n in aday.items()), default=0.0)
            if benzer >= ESIK or s in gorulen:
                atilan.append((round(benzer, 2), s))
                continue
            gorulen.add(s)
            kalan.append(s)
        f.write_text("\n".join(kalan) + "\n", encoding="utf-8", newline="\n")
        toplam_at += len(atilan)
        print(f"{f.stem:8s} {len(satir):4d} → {len(kalan):4d}  (atılan {len(atilan)}: {atilan[:2]})")
    print(f"toplam atılan {toplam_at}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
