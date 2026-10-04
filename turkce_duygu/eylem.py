# -*- coding: utf-8 -*-
"""
KONUŞMA EYLEMİ — cümlenin biçiminden ne yaptığını (soru, rica, uyarı, özür…) kurallarla çıkarır.

Duygu modelinin göremediği şeyi yakalar: "Bana yardım edebilir misiniz?" soru işaretli ama
merak değil, ricadır; "Dikkat, zemin ıslak." duygusuz görünür ama uyarıdır.

Kaynaklar:
  G&K   Göksel, A. & Kerslake, C. (2005) Turkish: A Comprehensive Grammar. Routledge.
        §19.1 evet/hayır soruları (mI), §19.2 ne-soruları, §19.5.2 acaba,
        §21.4.2.1 olasılık (-(y)Abil), §21.4.2.2 gereklilik (-mAlI; olumsuz gövdede yükümlülük
        yasaklamadır), §21.4.4.1 dilek (-sA, keşke), §21.4.4.2 emir, §21.4.4.3 öneri
        (-(y)AyIm, -(y)AlIm), §21.4.4.4 danışma (-sAm/-sAk + acaba), §21.4.4.5 rica ve teklif
        (2. kişi geniş zaman sorusu), §7.1.5 ünlemler.
  OC83  Olshtain, E. & Cohen, A. D. (1983) Apology: a speech act set. Wolfson & Judd (ed.),
        Sociolinguistics and Language Acquisition, 18–35. Özür stratejileri: açık özür sözü,
        sorumluluk üstlenme, açıklama, telafi önerisi, yinelememe sözü.
  H03   Hatipoğlu, Ç. (2003) Turkish apologies: özür dilerim en sık özür stratejisi; kusura bakma,
        üzgünüm hafif kusurlarda yaygın.
  TDK   Güncel Türkçe Sözlük: "dikkat" ünlem, "'Dikkat ediniz!' anlamında kullanılan bir uyarma
        sözü"; "sakın" ünlem, "asla"; merak, "bir şeyi öğrenmek, bilmek isteği"; ünlemlerin
        duygu tanımları (veri/unlemler.tsv).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

VERI = Path(__file__).resolve().parent / "veri"
UNLU = "aeıioöuüâîû"
_KALIN = set("aıouâû")
_YUVARLAK = set("oöuüû")

TURLER = ("ozur", "uyari", "yasak", "rica", "danisma", "soru", "dilek", "oneri", "emir", "unlem", "bildirim")


@dataclass
class Eylem:
    tur: str                                   # TURLER'den biri
    ipucu: list[str] = field(default_factory=list)   # kararı veren öğeler (açıklama için)
    unlem: str = ""                            # ünlemin TDK duygu grubu: sevinc, saskinlik, uzuntu, sikinti, korku


def kucuk(s: str) -> str:
    return unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i")).lower()


def sozcukler(cumle: str) -> list[str]:
    return re.findall(r"[a-zçğıöşüâîû]+(?:['’][a-zçğıöşüâîû]+)?", kucuk(cumle))


# ── Ses uyumu ───────────────────────────────────────────────────────────────────────────────
def _son_unlu(s: str) -> str:
    for c in reversed(s):
        if c in UNLU:
            return c
    return "e"


def _I(kok: str) -> str:
    v = _son_unlu(kok)
    return ("u" if v in _YUVARLAK else "ı") if v in _KALIN else ("ü" if v in _YUVARLAK else "i")


def _A(kok: str) -> str:
    return "a" if _son_unlu(kok) in _KALIN else "e"


def _yumusak(kok: str) -> str:
    """Ünlüyle başlayan ek öncesi: git→gid, et→ed, tat→tad (eylemlerde yalnız bunlar yumuşar)."""
    if kok.endswith("t") and (kok in ("git", "tat", "güt") or kok.endswith("et")):
        return kok[:-1] + "d"
    return kok


@lru_cache(maxsize=1)
def _fiiller() -> frozenset[str]:
    return frozenset(s.strip() for s in (VERI / "fiiller.txt").read_text(encoding="utf-8").splitlines()
                     if s.strip() and not s.startswith("#"))


def ekle(govde: str, ek: str) -> str:
    """Eki gövdeye uyumla ekler: A = a/e, I = ı/i/u/ü (her ünlü bir öncekine uyar), (y) kaynaştırma,
    ünlüyle başlayan ek öncesi git→gid. Örnek: ekle("konuş", "(y)AlIm") → "konuşalım"."""
    if ek.startswith("(y)"):
        ek = ek[3:]
        if govde[-1] in UNLU:
            govde += "y"
        else:
            govde = _yumusak(govde)
    s = govde
    for c in ek:
        s += _A(s) if c == "A" else _I(s) if c == "I" else c
    return s


# G&K §21.4.4.1–4: emir, 3. kişi emir, yasak (olumsuz emir), öneri (1. kişi istek), -sA (danışma/dilek)
_KIPLER = (("", "emir"), ("(y)In", "emir"), ("(y)InIz", "emir"), ("sAnA", "emir"), ("sAnIzA", "emir"),
           ("sIn", "emir3"), ("sInlAr", "emir3"),
           ("mA", "yasak"), ("mAyIn", "yasak"), ("mAyInIz", "yasak"), ("mAsIn", "yasak"), ("mAsInlAr", "yasak"),
           ("(y)AlIm", "oneri"), ("(y)AyIm", "oneri"), ("mAyAlIm", "oneri"),
           ("sAm", "danisma"), ("sAk", "danisma"))


@lru_cache(maxsize=1)
def _bicimler() -> dict[str, tuple[str, str]]:
    """Çekimli biçim → (eylem kipi, kök)."""
    b: dict[str, tuple[str, str]] = {}
    for k in _fiiller():
        for ek, kip in _KIPLER:
            b.setdefault(ekle(k, ek) if ek else k, (kip, k))
    return b


# Türetilmiş gövdeler (ettirgen, edilgen, işteş, dönüşlü): yanaş-tır, kullan-ıl, söyle-n, gör-üş.
_TUREME = re.compile(r"(?:d[ıiuü]r|t[ıiuü]r|[ıiuü]l|[ıiuü]n|[ıiuü]ş|[ae]r|t|l|n|ş)$")
_KIP_DESEN = [(re.compile("(" + ek.replace("(y)", "y?").replace("A", "[ae]").replace("I", "[ıiuü]") + ")$"), ek, kip)
              for ek, kip in _KIPLER if ek]


def _govde_mi(g: str, derin: int = 2) -> bool:
    if g in _fiiller() or (g.endswith("d") and g[:-1] + "t" in _fiiller()):
        return True
    m = _TUREME.search(g)
    return bool(derin and m and len(g) - len(m.group(0)) >= 2 and _govde_mi(g[:m.start()], derin - 1))


@lru_cache(maxsize=20000)
def fiil_kipi(sozcuk: str) -> str:
    """Sözcük bir emir/istek biçimiyse kipi ("emir", "emir3", "yasak", "oneri", "danisma"), değilse "".
    Önce sözlükteki eylemlerin çekimleri; sonra türetilmiş gövdeler (yanaştırmayın, kullanılmasın)."""
    if (b := _bicimler().get(sozcuk)):
        return b[0]
    for desen, ek, kip in _KIP_DESEN:
        if (m := desen.search(sozcuk)) and len(sozcuk) - len(m.group(1)) >= 3:
            g = sozcuk[:m.start(1)]
            if g and ekle(g, ek) == sozcuk and _govde_mi(g):
                return kip
    return ""


# G&K §21.4.2.2: gereklilik -mAlI; olumsuz gövdede (-mAmAlI) yasaklamadır.
_GEREK = re.compile(r"\w{3,}m[ae]l[ıi](?:s[ıi]n(?:[ıi]z)?|d[ıi]r|y[ıi]z|y[ıi]m|d[ıi]rl[ae]r)?$")
_OLUMSUZ_GEREK = re.compile(r"m[ae]m[ae]l[ıi](?:s[ıi]n(?:[ıi]z)?|d[ıi]r|y[ıi]z|y[ıi]m|d[ıi]rl[ae]r)?$")


def yuklem_kipi(sozcuk: str) -> str:
    """fiil_kipi + gereklilik: "girmemelisin" → yasak, "takmalıdır" → gerek."""
    if _OLUMSUZ_GEREK.search(sozcuk):
        return "yasak"
    if _GEREK.search(sozcuk):
        return "gerek"
    return fiil_kipi(sozcuk)


# ── Sözcük listeleri ────────────────────────────────────────────────────────────────────────
# H03: açık özür kalıpları; OC83: pişmanlık, sorumluluk üstlenme, telafi önerisi.
# "affedersiniz" soru önündeyse dikkat çekme sözüdür (aşağıda).
_OZUR = re.compile(r"\b(?:özür\s+dil(?:er|iyor)(?:im|iz|um|uz)|özür\s+dilemek\s+ist|kusura\s+bakma(?:y[ıi]n(?:[ıi]z)?)?|"
                   r"affeders[ıi]n(?:[ıi]z)?|affet(?:in|iniz)?|affola|bağışla(?:y[ıi]n(?:[ıi]z)?)?|"
                   r"üzgünüm|üzgünüz|pardon|kusurumuz|kusurum|"
                   r"pişman(?:ım|ız)|telafi\s+(?:ed|etm)\w*|haksızdım|haksızlık\s+ettim|"
                   r"(?:için|nedeniyle|dolayı)\s+üzülüyor(?:um|uz)|pişman(?:lık)?\s+(?:duyuyor|hissediyor)\w*|"
                   r"vicdan\s+azabı\s+duy\w*|hata(?:mı|mızı)\s+kabul\s+ed\w*|hata\s+yapt(?:ım|ık))\b")
# TDK: "dikkat" ünlemi uyarma sözüdür; "dikkat çek-" (ilgi toplamak) uyarı değildir.
_DIKKAT = re.compile(r"(?:^|[,;:!.]\s*)dikkat\b(?!\s+çek)|\bdikkat(?:li)?\s+(?:et|edin|ediniz|edilmeli|olun|olunuz|ol)\b|"
                     r"\bdikkatli\s+\w+(?:[ıiuü]n|[ıiuü]n[ıiuü]z)\b")
_SAKIN = re.compile(r"(?:^|[,;:!.]\s*)(?:sakın|aman)\b")
_ASLA = re.compile(r"\b(?:asla|sakın|kesinlikle|hiçbir\s+şekilde)\b")
# Tehlike bildiren sözcükler: buyrukla ya da bir eylem adıyla (-mAk) birlikte uyarıdır; tek başına haber olabilir.
_TEHLIKE = re.compile(r"\b(?:tehlike\w*|risk\w*|kaygan|buzlanma|zehirli|yanıcı|patlayıcı|yüksek\s+gerilim|"
                      r"kaza\w*|yaralan\w*|çökme\w*|yangın\w*|zarar\w*|güvenliğ\w*|ceza\w*|yasak\w*|uyarı\w*|alarm\w*)\b")
# Olası kötü sonuç (G&K §21.4.2.1 olasılık): buyrukla birlikte "…atma, yangın çıkabilir"; "aksi halde", "yoksa".
_SONUC = re.compile(r"\w+[ae]bil(?:ir|irsin|irsiniz|irler)\b|\b(?:aksi\s+(?:halde|takdirde)|yoksa)\b")
_EYLEM_ADI = re.compile(r"\w{2,}m[ae]k\b|\w{2,}m[ae]s[ıi]\b")
_SORU_EKI = re.compile(r"^m[ıiuü](?:y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|d[ıiuü]r|y[ıiuü]d[ıiuü]\w*|"
                       r"y[ıiuü]ş|y[ıiuü]m[ıiuü]ş\w*|yken|l[ae]r)?$")
# Yüklemden sonra gelebilen sözler (devrik cümle, seslenme): "hemen bırak onu", "gel artık".
_SON_DISI = {"acaba", "lütfen", "yahu", "be", "ya", "hadi", "haydi", "onu", "bunu", "şunu", "hemen", "artık"}
_BELIRTEC = {"bir", "bu", "şu", "o", "her", "çok", "en", "ne", "hiç"}   # ardındaki çıplak kök addır: "bu bir at"
_IKINCI_KISI = re.compile(r"^m[ıiuü]s[ıiuü]n(?:[ıiuü]z)?$")
_BIRINCI_KISI = re.compile(r"^m[ıiuü]y[ıiuü][mz]$")
# TDK merak: "bir şeyi öğrenmek, bilmek isteği". Konuşanın kendi merakını söylemesi soru işlevi görür.
_MERAK = re.compile(r"\bmerak\s+(?:ediyor(?:um|uz)|ettim|ederim|içindeyim)\b")


@lru_cache(maxsize=1)
def _unlemler() -> list[tuple[re.Pattern, str]]:
    """(desen, grup); iki sözcüklüler önce ("vah vah" "vah"tan, "oh be" "oh"tan önce denenir)."""
    satir = [s.split("\t") for s in (VERI / "unlemler.tsv").read_text(encoding="utf-8").splitlines()
             if s and not s.startswith("#")]
    satir.sort(key=lambda p: -len(p[0]))
    return [(re.compile(rf"^{re.escape(u)}(?:\s*[,!.…]|$)" if konum == "bas" else rf"\b{re.escape(u)}\b"), grup)
            for u, grup, konum, *_ in satir]


def _unlem_bul(k: str) -> str:
    for desen, grup in _unlemler():
        if desen.search(k):
            return grup
    return ""


def _yan_cumle_sonlari(k: str) -> list[str]:
    """Her yan cümlenin (virgül, noktalı virgül, iki nokta) son sözcüğü: "kibritle oynama, yanarsın"."""
    sonuc = []
    for parca in re.split(r"[,;:]", k):
        ss = [s for s in sozcukler(parca) if not _SORU_EKI.match(s) and s not in _SON_DISI]
        if ss:
            sonuc.append(ss[-1])
    return sonuc


# ── Karar ───────────────────────────────────────────────────────────────────────────────────
def eylem(cumle: str) -> Eylem:
    k = kucuk(cumle).strip()
    ss = sozcukler(cumle)
    if not ss:
        return Eylem("bildirim")
    son = next((s for s in reversed(ss) if not _SORU_EKI.match(s) and s not in _SON_DISI), ss[-1])
    soru_ekleri = [i for i, s in enumerate(ss) if _SORU_EKI.match(s)]
    soru = k.rstrip().endswith("?") or "?" in k[-3:] or bool(soru_ekleri)
    unlem = _unlem_bul(k)

    # 1) Özür (H03, OC83). "Affedersiniz, … ?" bir dikkat çekme sözüdür → soru kuralları karar verir.
    m = _OZUR.search(k)
    if m and not (soru and m.group(0).startswith(("affeders", "pardon"))):
        return Eylem("ozur", [m.group(0)], unlem)

    kip = yuklem_kipi(son)
    if kip == "yasak" and len(son) <= 4 and not (k.endswith("!") or _ASLA.search(k)):
        kip = ""                                                   # "aşma", "atma" ad da olabilir
    yan = [x for x in (yuklem_kipi(s) for s in _yan_cumle_sonlari(k)[:-1]) if x]
    buyruk = next((x for x in [kip] + yan if x in ("emir", "emir3", "yasak", "gerek")), "")

    # 2) Uyarı: dikkat (TDK); sakın/aman + buyruk; buyruk + tehlike ya da olası kötü sonuç;
    #    eylem adı + tehlike ("…yürümek çok tehlikeli").
    if (m := _DIKKAT.search(k)):
        return Eylem("uyari", [m.group(0).strip(" ,;:!.")], unlem)
    if not soru:
        if (m := _SAKIN.search(k) or (buyruk == "yasak" and _ASLA.search(k))) and (buyruk or k.endswith("!")):
            return Eylem("uyari", [m.group(0).strip(" ,;:!."), son], unlem)
        if buyruk and (m := _TEHLIKE.search(k) or _SONUC.search(k)):
            return Eylem("uyari", [m.group(0), buyruk], unlem)
        if _EYLEM_ADI.search(k) and (m := _TEHLIKE.search(k)):
            return Eylem("uyari", [m.group(0), "-mAk"], unlem)

    # 3) Sorular (G&K §19, §21.4.4.3–5).
    if soru:
        if soru_ekleri:
            i = soru_ekleri[-1]
            once = ss[i - 1] if i else ""
            genis = once.endswith(("r", "z")) and not once.endswith("yor")    # geniş zaman; -Iyor değil
            if _IKINCI_KISI.match(ss[i]) and genis:
                return Eylem("rica", [once, ss[i]], unlem)            # gelir misin, edebilir misiniz, oturmaz mısınız
            if _BIRINCI_KISI.match(ss[i]) and once.endswith("bilir"):
                return Eylem("rica", [once, ss[i]], unlem)            # sorabilir miyim (izin isteme)
            if fiil_kipi(once) in ("oneri", "danisma"):
                return Eylem("danisma", [once, ss[i]], unlem)         # yardım edeyim mi, gidelim mi
        if "lütfen" in ss or re.search(r"\brica\s+ets", k):
            return Eylem("rica", ["lütfen" if "lütfen" in ss else "rica"], unlem)
        if fiil_kipi(son) in ("danisma", "oneri", "emir3"):
            # kime danışsam acaba (§21.4.4.4), nereye koyalım (§21.4.4.3), burada mı kalsın (§21.4.4.2)
            return Eylem("danisma", [son] + (["acaba"] if "acaba" in ss else []), unlem)
        return Eylem("soru", ["acaba"] if "acaba" in ss else ([ss[soru_ekleri[-1]]] if soru_ekleri else ["?"]), unlem)
    if (m := _MERAK.search(k)):
        return Eylem("soru", [m.group(0)], unlem)

    # 4) Dilek (G&K §21.4.4.1): keşke; soru olmayan 1. kişi -sA.
    if "keşke" in ss:
        return Eylem("dilek", ["keşke"], unlem)
    if kip == "danisma":
        return Eylem("dilek", [son], unlem)                       # bu akşam film seyretsek (§21.4.4.1)
    # 5) Yasak / öneri / emir (G&K §21.4.4.2–3, §21.4.2.2). Tek başına çıplak kök, belirteçten
    #    sonra değilse emirdir ("Bu bir at." gibi adlarla karışmasın).
    if kip == "yasak":
        return Eylem("yasak", [son], unlem)
    if kip == "oneri":
        return Eylem("oneri", [son], unlem)
    if kip in ("emir", "emir3", "gerek"):
        ciplak = son == ss[-1] and son in _fiiller()
        if not (ciplak and len(ss) > 1 and ss[-2] in _BELIRTEC):
            return Eylem("rica" if "lütfen" in ss else "emir", [son], unlem)
    # 6) Ünlem.
    if unlem or k.endswith("!"):
        return Eylem("unlem", ["!"] if not unlem else [unlem], unlem)
    return Eylem("bildirim", [], unlem)
