# -*- coding: utf-8 -*-
"""
TÜRKÇE DUYGU — Türkçe metinden seslendirme stili: konuşma eylemi (kurallı) + duygu (model).

    from turkce_duygu import cozumle, metin_cozumle
    r = cozumle("Dikkat, zemin ıslak!")
    r.eylem, r.stil, r.yogunluk          # 'uyari', 'uyari', 0.95
    for r in metin_cozumle("Günaydın! Bugün hava çok güzel. Şemsiyeni unutma, yağmur yağabilir."):
        print(r.cumle, r.stil, r.yogunluk)

Model (duygu.onnx) yoksa yalnız eylem kuralları çalışır; stil yine soru/uyarı/özür/ünlemden seçilir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .eylem import Eylem, eylem
from .stil import STILLER, Stil, stil_sec, yumusat

__all__ = ["Cozumleyici", "Sonuc", "cozumle", "metin_cozumle", "eylem", "cumlelere_bol", "STILLER"]
__version__ = "0.1.1"


@dataclass
class Sonuc:
    cumle: str
    eylem: str
    ipucu: list[str]
    stil: str
    yogunluk: float
    gerekce: str
    duygular: dict[str, float] = field(default_factory=dict)   # Plutchik 8 duygu olasılığı (model varsa)


_KISALTMA = ("Dr.", "Prof.", "Doç.", "Yrd.", "Av.", "Müh.", "Sn.", "vb.", "vs.", "bkz.", "örn.", "St.",
             "Cad.", "Sok.", "Mah.", "No.", "Apt.")


try:                                    # kardeş proje kuruluysa onun bölücüsü (geniş kısaltma listesi)
    from turkce_normalize import cumleleri_ayir as _normalize_bol
except ImportError:                     # pragma: no cover
    _normalize_bol = None


def cumlelere_bol(metin: str) -> list[str]:
    """Cümlelere böler. turkce-normalize kuruluysa onun bölücüsü (seslendirmeyle aynı bölme);
    yoksa basit bölücü: . ! ? … ardından boşluk ve büyük harf/rakam/tırnak; kısaltmada bölmez."""
    if _normalize_bol is not None:
        return [c for c in _normalize_bol(metin, 0) if c.strip()]
    parcalar, bas = [], 0
    for m in re.finditer(r"[.!?…]+[\"”’»)]*\s+(?=[\"“‘«(]*[A-ZÇĞİÖŞÜ0-9])", metin):
        onceki = metin[bas:m.end()].rstrip()
        if any(onceki.endswith(k) for k in _KISALTMA):
            continue
        parcalar.append(onceki)
        bas = m.end()
    parcalar.append(metin[bas:])
    return [p.strip() for p in parcalar if p.strip()]


class Cozumleyici:
    def __init__(self, model_klasoru: str | None = None, model: bool | None = None,
                 stiller: tuple[str, ...] = STILLER, ayar: dict | None = None):
        """model=None: varsa yükle; True: zorunlu; False: yalnız kurallar.
        stiller: seslendirecek sesin sahip olduğu stiller (olmayanlar en yakınına düşer).
        ayar: stil.VARSAYILAN_AYAR anahtarlarını ezer (model klasöründeki ayarın da üstünde)."""
        self.stiller = stiller
        self.ayar = ayar
        self.model = None
        if model is not False:
            try:
                from .model import DuyguModeli
                self.model = DuyguModeli(model_klasoru)
            except (FileNotFoundError, ImportError):
                if model:
                    raise

    def _gucler(self, cumleler: list[str]) -> tuple[list[dict | None], list[dict], list[dict | None]]:
        if not self.model or not cumleler:
            return [None] * len(cumleler), [{} for _ in cumleler], [None] * len(cumleler)
        p, sp = self.model.tahmin(cumleler)
        g = self.model.guc(p)
        d = self.model.duygular
        so = [dict(zip(self.model.stiller, map(float, r))) for r in sp] if sp is not None else [None] * len(cumleler)
        return ([dict(zip(d, map(float, r))) for r in g],
                [dict(zip(d, (round(float(x), 4) for x in r))) for r in p], so)

    def cozumle(self, cumle: str) -> Sonuc:
        return self.metin_cozumle(cumle, bol=False)[0]

    def metin_cozumle(self, metin: str, bol: bool = True, surekli: bool = True) -> list[Sonuc]:
        cumleler = cumlelere_bol(metin) if bol else [metin.strip()]
        gucler, olas, stil_olas = self._gucler(cumleler)
        eylemler: list[Eylem] = [eylem(c) for c in cumleler]
        ayar = {**(getattr(self.model, "stil_ayar", None) or {}), **(self.ayar or {})}
        stiller: list[Stil] = [stil_sec(c, e, g, self.stiller, ayar, so)
                               for c, e, g, so in zip(cumleler, eylemler, gucler, stil_olas)]
        if surekli:
            stiller = yumusat(stiller, gucler)
        return [Sonuc(c, e.tur, e.ipucu, s.stil, s.yogunluk, s.gerekce, o)
                for c, e, s, o in zip(cumleler, eylemler, stiller, olas)]


_VARSAYILAN: Cozumleyici | None = None


def _varsayilan() -> Cozumleyici:
    global _VARSAYILAN
    if _VARSAYILAN is None:
        _VARSAYILAN = Cozumleyici()
    return _VARSAYILAN


def cozumle(cumle: str) -> Sonuc:
    return _varsayilan().cozumle(cumle)


def metin_cozumle(metin: str, surekli: bool = True) -> list[Sonuc]:
    return _varsayilan().metin_cozumle(metin, surekli=surekli)
