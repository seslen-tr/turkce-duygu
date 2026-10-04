# -*- coding: utf-8 -*-
"""
KONUŞMA STİLİ — konuşma eylemi + duygu olasılıkları → seslendirme stili ve yoğunluğu.

Stiller: nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın, fısıltı. Fısıltı metinden seçilmez
(anlatıcı/stüdyo kararı); yalnız "fısıldadı" gibi açık söz geçerse önerilir. Sesin bir stili yoksa
YEDEK'teki en yakın stile düşülür (ör. üzgün → özür).
Önce biçim karar verir (soru → merak, özür kalıbı → özür, uyarı → uyarı; G&K ve H03'e dayanan
eylem.py), biçimin söylemediği yerde duygu modeli. Duygu → stil eşlemesi aşağıda, değiştirilebilir.
Yoğunluk 0–1: stüdyoda duygu yoğunluğu sürgüsüne verilir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .eylem import Eylem, kucuk

STILLER = ("notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin", "fisilti")
YEDEK = {"uzgun": "ozur", "kizgin": "uyari", "fisilti": "notr"}

# Plutchik duygusu → stil. Korku için ayrı stil yok; en yakını gergin, uyaran ton.
DUYGU_STIL = {"sevinc": "neseli", "guven": "neseli", "saskinlik": "heyecan", "bekleyis": "heyecan",
              "korku": "uyari", "ofke": "kizgin", "tiksinti": "kizgin", "uzuntu": "uzgun"}
# Ünlemin TDK duygu grubu → stil
UNLEM_STIL = {"sevinc": "neseli", "saskinlik": "heyecan", "uzuntu": "uzgun", "korku": "uyari", "sikinti": "kizgin"}
_FISILTI = re.compile(r"\b(?:fısılda\w*|fısıltı\w*|kulağına\s+eğil\w*)\b")
# Eylemin kendi stili ve taban yoğunluğu (None: duygu modeline bırakılır)
EYLEM_STIL = {"ozur": ("ozur", 0.7), "uyari": ("uyari", 0.8), "yasak": ("uyari", 0.6), "emir": ("uyari", 0.3),
              "soru": ("merak", 0.6), "danisma": ("merak", 0.45), "rica": ("notr", 0.0)}
# Stil ayarı (model klasöründeki esikler.json "stil" anahtarı ezer; degerlendirme/stil_ayar.py seçer):
#   duyarlilik: duygu gücü çarpanı (büyükse daha az cümle nötr kalır)
#   baskin: biçimi ezen duygu gücü — "Bunu bana nasıl yaparsın?" soru biçiminde ama öfkeliyse kızgın okunur
VARSAYILAN_AYAR = {"duyarlilik": 1.0, "baskin": 1.5}
_PEKISTIRME = re.compile(r"\b(?:çok|mutlaka|kesinlikle|asla|hemen|derhal|sakın|gerçekten|son\s+derece|inanılmaz\w*)\b")


@dataclass
class Stil:
    stil: str
    yogunluk: float
    gerekce: str


def _pekistir(cumle: str, taban: float) -> float:
    """Ünlem işareti ve pekiştirme sözcükleri yoğunluğu artırır."""
    k = kucuk(cumle)
    artis = 0.15 * min(2, len(_PEKISTIRME.findall(k))) + (0.15 if cumle.rstrip().endswith("!") else 0.0)
    if sum(c.isupper() for c in cumle) > 0.6 * max(1, sum(c.isalpha() for c in cumle)) and len(cumle) > 8:
        artis += 0.15                                            # BÜYÜK HARFLE yazılmış
    return round(min(1.0, taban + artis), 2)


def stil_sec(cumle: str, e: Eylem, guc: dict[str, float] | None, mevcut: tuple[str, ...] = STILLER,
             ayar: dict | None = None) -> Stil:
    """guc: duygu başına (olasılık / eşik) / nötr eşiği; >1 ise o duygu belirgin. None: model yok.
    mevcut: sesin sahip olduğu stiller; olmayan stil YEDEK'e düşer."""
    a = {**VARSAYILAN_AYAR, **(ayar or {})}
    if guc:
        guc = {d: g * a["duyarlilik"] for d, g in guc.items()}
    s = _stil_sec(cumle, e, guc, a["baskin"])
    while s.stil not in mevcut and s.stil in YEDEK:
        s = Stil(YEDEK[s.stil], s.yogunluk if YEDEK[s.stil] != "notr" else 0.0, s.gerekce + f" (yedek: {s.stil}→{YEDEK[s.stil]})")
    return s


def _stil_sec(cumle: str, e: Eylem, guc: dict[str, float] | None, baskin: float = 1.5) -> Stil:
    if _FISILTI.search(kucuk(cumle)):
        return Stil("fisilti", 0.6, "fısıltı sözü")
    if guc and e.tur in ("soru", "danisma", "emir", "yasak", "uyari", "unlem"):
        # Güçlü duygu biçimi ezer: öfkeli soru / emir → kızgın, hüzünlü soru → üzgün
        ofke = max(guc.get("ofke", 0.0), guc.get("tiksinti", 0.0))
        if ofke >= baskin:
            return Stil("kizgin", _pekistir(cumle, float(np.clip(0.4 + (ofke - baskin) / 2, 0.4, 0.9))),
                        f"öfke ({ofke:.2f}) biçimi ({e.tur}) ezdi")
        if e.tur in ("soru", "danisma") and guc.get("uzuntu", 0.0) >= baskin:
            return Stil("uzgun", _pekistir(cumle, 0.5), f"üzüntü ({guc['uzuntu']:.2f}) biçimi ({e.tur}) ezdi")
    if e.tur in EYLEM_STIL:
        s, y = EYLEM_STIL[e.tur]
        # "Gerçekten mi?!" gibi: biçim soru, duygu şaşkınlık → heyecan
        if s == "merak" and guc and "!" in cumle[-3:] and guc.get("saskinlik", 0) > 1:
            return Stil("heyecan", _pekistir(cumle, 0.6), "soru + ünlem + şaşkınlık")
        return Stil(s, _pekistir(cumle, y) if y else 0.0, f"eylem: {e.tur} ({', '.join(e.ipucu)})")
    if e.unlem in UNLEM_STIL:
        return Stil(UNLEM_STIL[e.unlem], _pekistir(cumle, 0.6), f"ünlem: {e.unlem}")
    if guc:
        stil_guc: dict[str, float] = {}
        for d, g in guc.items():
            s = DUYGU_STIL[d]
            stil_guc[s] = max(stil_guc.get(s, 0.0), g)
        s, g = max(stil_guc.items(), key=lambda x: x[1])
        if g >= 1.0:
            d = max((d for d in guc if DUYGU_STIL[d] == s), key=guc.get)
            return Stil(s, _pekistir(cumle, float(np.clip(0.3 + (g - 1.0) / 1.5, 0.3, 0.85))), f"duygu: {d} ({g:.2f})")
    if cumle.rstrip().endswith("!"):
        return Stil("heyecan", 0.4, "ünlem işareti")
    return Stil("notr", 0.0, "belirgin duygu yok" if guc else "model yok")


def yumusat(stiller: list[Stil], gucler: list[dict[str, float] | None]) -> list[Stil]:
    """Paragraf içinde ani stil kopmalarını azaltır: duygulu bir cümlenin ardından gelen nötr cümle,
    aynı duyguya eşiğe yakın (≥0.7) bir eğilim gösteriyorsa önceki stili yarı yoğunlukla sürdürür."""
    sonuc = list(stiller)
    for i in range(1, len(sonuc)):
        onceki, simdi, g = sonuc[i - 1], sonuc[i], gucler[i]
        if simdi.stil == "notr" and onceki.stil not in ("notr", "merak") and g and onceki.yogunluk > 0:
            yakin = max((v for d, v in g.items() if DUYGU_STIL[d] == onceki.stil), default=0.0)
            if yakin >= 0.7:
                sonuc[i] = Stil(onceki.stil, round(onceki.yogunluk * 0.5, 2), f"süreklilik ({yakin:.2f})")
    return sonuc
