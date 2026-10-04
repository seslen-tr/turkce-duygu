# -*- coding: utf-8 -*-
"""
EĞİTİM — BERTurk'ü (dbmdz/bert-base-turkish-cased, MIT) XED Türkçe üzerinde çok etiketli duygu
sınıflandırmasına ince ayarlar. İşlemcide çalışır (GPU gerekmez; ~1 saat).

    python egit/veri.py && python egit/egit.py

Çıktı model/hf/: ağırlıklar + belirteçleyici + esikler.json (doğrulamada seçilen duygu başına eşik
ve nötr eşiği). En iyi tur doğrulama makro-F1'ine göre seçilir; sınav kümesine burada bakılmaz.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.veri import DUYGULAR  # noqa: E402
from turkce_duygu.model import hazirla  # noqa: E402

TABAN = "dbmdz/bert-base-turkish-cased"
UZUNLUK = 64


def oku(ad: str) -> tuple[list[str], np.ndarray]:
    metin, y = [], []
    for s in (KOK / "veri" / f"xed_{ad}.tsv").read_text(encoding="utf-8").splitlines():
        c, e = s.split("\t")
        metin.append(hazirla(c))
        y.append([float(d in e.split(",")) for d in DUYGULAR])
    return metin, np.array(y, dtype=np.float32)


def notr_oku(ad: str) -> list[str]:
    return [hazirla(s) for s in (KOK / "veri" / f"notr_{ad}.txt").read_text(encoding="utf-8").splitlines() if s]


def notr_esik_sec(p_duygu: np.ndarray, p_notr: np.ndarray, esik: np.ndarray) -> tuple[float, float]:
    """Nötr eşiği: duygulu (XED) cümlelerin duygulu, nötr cümlelerin nötr çıkma oranlarının ortalamasını
    (dengeli doğruluk) en yükselten ölçek. Döner (eşik, dengeli doğruluk)."""
    gd, gn = (p_duygu / esik).max(1), (p_notr / esik).max(1)
    adaylar = np.quantile(np.concatenate([gd, gn]), np.linspace(0.02, 0.98, 97))
    puan = [((gd >= a).mean() + (gn < a).mean()) / 2 for a in adaylar]
    i = int(np.argmax(puan))
    return float(adaylar[i]), float(puan[i])


def f1(y: np.ndarray, p: np.ndarray, esik: np.ndarray) -> np.ndarray:
    t = p >= esik
    dp = (t & (y > 0)).sum(0)
    return 2 * dp / np.maximum(t.sum(0) + (y > 0).sum(0), 1)


def esik_sec(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Duygu başına doğrulama F1'ini en yükselten eşik."""
    adaylar = np.linspace(0.05, 0.9, 35)
    return np.array([max(adaylar, key=lambda e: f1(y[:, [j]], p[:, [j]], np.array([e]))[0]) for j in range(y.shape[1])])


@torch.no_grad()
def tahmin(model, tok, metin: list[str], parti: int = 64) -> np.ndarray:
    model.eval()
    cikti = []
    for i in range(0, len(metin), parti):
        g = tok(metin[i:i + parti], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt")
        cikti.append(torch.sigmoid(model(**g).logits).numpy())
    return np.concatenate(cikti)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tur", type=int, default=4)
    ap.add_argument("--parti", type=int, default=32)
    ap.add_argument("--oran", type=float, default=3e-5)
    ap.add_argument("--adim-sinir", type=int, default=0, help="süre ölçümü için: bu kadar adımdan sonra dur")
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(TABAN)
    model = AutoModelForSequenceClassification.from_pretrained(
        TABAN, num_labels=len(DUYGULAR), problem_type="multi_label_classification",
        id2label=dict(enumerate(DUYGULAR)), label2id={d: i for i, d in enumerate(DUYGULAR)})
    xe, ye = oku("egitim")
    xd, yd = oku("dogrulama")
    ne = notr_oku("egitim")                       # nötr: etiketsiz (hepsi 0) satır olarak eğitime girer
    xe, ye = xe + ne, np.concatenate([ye, np.zeros((len(ne), len(DUYGULAR)), dtype=np.float32)])
    notr = notr_oku("dogrulama")

    adim_tur = math.ceil(len(xe) / a.parti)
    opt = torch.optim.AdamW(model.parameters(), lr=a.oran, weight_decay=0.01)
    plan = get_linear_schedule_with_warmup(opt, int(0.1 * adim_tur * a.tur), adim_tur * a.tur)
    hedef = KOK / "model" / "hf"
    en_iyi, adim = -1.0, 0
    for tur in range(a.tur):
        model.train()
        sira = list(range(len(xe)))
        random.shuffle(sira)
        t0, kayip_top = time.time(), 0.0
        for b in range(adim_tur):
            idx = sira[b * a.parti:(b + 1) * a.parti]
            g = tok([xe[i] for i in idx], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt")
            kayip = model(**g, labels=torch.from_numpy(ye[idx])).loss
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            plan.step()
            opt.zero_grad()
            kayip_top += kayip.item()
            adim += 1
            if b % 20 == 0:
                print(f"tur {tur} adım {b}/{adim_tur} kayıp {kayip_top / (b + 1):.4f} "
                      f"({(time.time() - t0) / (b + 1):.2f} sn/adım)", flush=True)
            if a.adim_sinir and adim >= a.adim_sinir:
                return 0
        pd = tahmin(model, tok, xd)
        esik = esik_sec(yd, pd)
        makro = float(f1(yd, pd, esik).mean())
        ilk = float(np.mean(yd[np.arange(len(yd)), pd.argmax(1)] > 0))
        print(f"== tur {tur}: doğrulama makro-F1 {makro:.4f}, ilk tahmin doğruluğu {ilk:.4f}", flush=True)
        if makro > en_iyi:
            en_iyi = makro
            pn = tahmin(model, tok, notr)
            notr_esik, dengeli = notr_esik_sec(pd, pn, esik)
            hedef.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(hedef)
            tok.save_pretrained(hedef)
            (hedef / "esikler.json").write_text(json.dumps(
                {"duygular": list(DUYGULAR), "esik": [round(float(e), 4) for e in esik], "notr_esik": round(notr_esik, 4),
                 "tur": tur, "dogrulama_makro_f1": round(makro, 4), "dogrulama_ilk": round(ilk, 4),
                 "dogrulama_notr_dengeli": round(dengeli, 4)},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"   kaydedildi ({hedef}); nötr eşiği {notr_esik:.3f} (duygulu/nötr dengeli doğruluk {dengeli:.3f})",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
