# -*- coding: utf-8 -*-
"""python -m turkce_duygu "Metin..."  — her cümle için eylem, stil, yoğunluk ve gerekçe."""
import sys

from . import metin_cozumle


def main() -> int:
    metin = " ".join(sys.argv[1:]) or sys.stdin.read()
    for r in metin_cozumle(metin):
        enguclu = max(r.duygular, key=r.duygular.get) if r.duygular else "-"
        print(f"{r.stil:8s} {r.yogunluk:.2f}  [{r.eylem}; {r.gerekce}; en olası duygu: {enguclu}]  {r.cumle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
