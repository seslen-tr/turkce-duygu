# -*- coding: utf-8 -*-
"""
STİL BAŞLIĞI — duygu modelinin (model/hf) gövdesine ikinci bir çıkış ekler: 8 seslendirme stili
(nötr, neşeli, heyecan, uyarı, merak, özür, üzgün, kızgın). İki çıkış birlikte eğitilir:

  duygu  8 Plutchik duygusu (sigmoid)   XED Türkçe + nötr örnekler (hepsi 0)
  stil   8 stil (softmax)               açık stil seti (turkce_duygu/veri/stil_seti, CC0, bu proje için
                                        yazıldı) + nötr örnekler (nötr) + XED (duygudan eşlenmiş stil,
                                        düşük ağırlıkla: eşleme yaklaşıktır)

Model seçimi: Seslen havuzlarının İLK yarısında (geliştirme) sınıf dengeli stil doğruluğu; havuz yoksa
XED doğrulama makro-F1. Sınav (ikinci yarılar) degerlendirme/stil_havuz.py ile.
Çıktı: model/onnx/duygu.onnx (iki çıkış: logits, stil_logits) — PyTorch ile eşdeğerliği burada denetlenir.

    python egit/egit_stil.py            (eğit + aktar)
    python egit/egit_stil.py aktar      (model/stil_gecici/model.pt'den yalnız ONNX'e aktar)
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import f1, notr_oku, oku  # noqa: E402
from egit.veri import DUYGULAR  # noqa: E402
from turkce_duygu.model import hazirla  # noqa: E402

STILLER = ("notr", "neseli", "heyecan", "uyari", "merak", "ozur", "uzgun", "kizgin")
ESLEME = {"sevinc": "neseli", "guven": "neseli", "saskinlik": "heyecan", "bekleyis": "heyecan",
          "ofke": "kizgin", "tiksinti": "kizgin", "uzuntu": "uzgun", "korku": "uyari"}
HAVUZ = Path(os.environ.get("TURKCE_DUYGU_HAVUZ", Path(__file__).resolve().parents[1] / "veri" / "havuz"))
UZUNLUK = 64


class IkiCikis(nn.Module):
    def __init__(self, govde, duygu_basi):
        super().__init__()
        self.govde, self.duygu = govde, duygu_basi
        self.birak = nn.Dropout(0.1)
        self.stil = nn.Linear(govde.config.hidden_size, len(STILLER))

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.birak(self.govde(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids).pooler_output)
        return self.duygu(h), self.stil(h)


def stil_seti() -> list[tuple[str, int]]:
    out = []
    for i, s in enumerate(STILLER):
        f = KOK / "turkce_duygu" / "veri" / "stil_seti" / f"{s}.txt"
        out += [(hazirla(c), i) for c in f.read_text(encoding="utf-8").splitlines() if c.strip()]
    return out


def gelistirme() -> list[tuple[str, int]]:
    out = []
    for i, s in enumerate(STILLER):
        f = HAVUZ / f"{s}.txt"
        if s == "notr" or not f.exists():
            continue
        ss = [c for c in f.read_text(encoding="utf-8").splitlines() if c.strip()]
        out += [(hazirla(c), i) for c in ss[:len(ss) // 2]]
    korpus = HAVUZ.parent / "korpus.jsonl"
    if korpus.exists():
        cum = [json.loads(x)["metin"] for x in korpus.read_text(encoding="utf-8").splitlines()]
        out += [(hazirla(c), 0) for c in [x for x in cum if x.endswith(".")][0::40][:400]]
    return out


@torch.no_grad()
def tahmin(model, tok, metin, parti=64):
    model.eval()
    d, s = [], []
    for i in range(0, len(metin), parti):
        g = tok(metin[i:i + parti], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt")
        a, b = model(g["input_ids"], g["attention_mask"], g["token_type_ids"])
        d.append(torch.sigmoid(a).numpy())
        s.append(torch.softmax(b, -1).numpy())
    return np.concatenate(d), np.concatenate(s)


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    random.seed(0)
    kaynak = KOK / "model" / "hf"
    tok = AutoTokenizer.from_pretrained(kaynak)
    taban = AutoModelForSequenceClassification.from_pretrained(kaynak)
    model = IkiCikis(taban.bert, taban.classifier)
    ayar = json.loads((kaynak / "esikler.json").read_text(encoding="utf-8"))
    esik = np.array(ayar["esik"])

    # (metin, duygu vektörü ya da None, stil sınıfı ya da -1, stil ağırlığı)
    ornek = []
    xe, ye = oku("egitim")
    for c, y in zip(xe, ye):
        ilk = [DUYGULAR[j] for j in np.flatnonzero(y)]
        ornek.append((c, y, STILLER.index(ESLEME[ilk[0]]) if len(ilk) == 1 else -1, 0.3))
    ornek += [(c, np.zeros(len(DUYGULAR), np.float32), 0, 1.0) for c in notr_oku("egitim")]
    ss = stil_seti()
    ornek += [(c, None, k, 1.0) for c, k in ss] * 3          # küçük ama alana uygun: 3 kez
    xd, yd = oku("dogrulama")
    gel = gelistirme()
    print(f"eğitim {len(ornek)} satır (stil seti {len(ss)} × 3), geliştirme {len(gel)}", flush=True)

    parti, tur = 32, 3
    adim_tur = math.ceil(len(ornek) / parti)
    opt = torch.optim.AdamW([{"params": model.govde.parameters(), "lr": 2e-5},
                             {"params": list(model.duygu.parameters()) + list(model.stil.parameters()), "lr": 1e-4}],
                            weight_decay=0.01)
    plan = get_linear_schedule_with_warmup(opt, int(0.06 * adim_tur * tur), adim_tur * tur)
    bce, ce = nn.BCEWithLogitsLoss(reduction="none"), nn.CrossEntropyLoss(reduction="none")
    hedef = KOK / "model" / "stil_gecici"
    en_iyi = -1.0
    if "aktar" in sys.argv[1:]:
        tur = 0
        model.load_state_dict(torch.load(hedef / "model.pt"))
        _, ps = tahmin(model, tok, [c for c, _ in gel])
        en_iyi = float(np.mean([np.mean([ps[i].argmax() == k for i, (_, k) in enumerate(gel) if k == s])
                                for s in sorted({k for _, k in gel})]))
        print(f"kayıtlı ağırlık: geliştirme stil (sınıf dengeli) %{100 * en_iyi:.1f}", flush=True)
    for t in range(tur):
        model.train()
        random.shuffle(ornek)
        t0 = time.time()
        for b in range(adim_tur):
            p = ornek[b * parti:(b + 1) * parti]
            g = tok([x[0] for x in p], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt")
            d_logit, s_logit = model(g["input_ids"], g["attention_mask"], g["token_type_ids"])
            dm = torch.tensor([x[1] is not None for x in p], dtype=torch.float32)
            dy = torch.tensor(np.stack([x[1] if x[1] is not None else np.zeros(len(DUYGULAR), np.float32) for x in p]))
            kd = (bce(d_logit, dy).mean(1) * dm).sum() / dm.sum().clamp(min=1)
            sy = torch.tensor([max(x[2], 0) for x in p])
            sw = torch.tensor([x[3] if x[2] >= 0 else 0.0 for x in p])
            ks = (ce(s_logit, sy) * sw).sum() / sw.sum().clamp(min=1e-6)
            kayip = kd + ks
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            plan.step()
            opt.zero_grad()
            if b % 50 == 0:
                print(f"tur {t} adım {b}/{adim_tur} duygu {kd.item():.3f} stil {ks.item():.3f} "
                      f"({(time.time() - t0) / (b + 1):.2f} sn/adım)", flush=True)
        pd_, _ = tahmin(model, tok, xd)
        makro = float(f1(yd, pd_, esik).mean())
        if gel:
            _, ps = tahmin(model, tok, [c for c, _ in gel])
            dogru = [np.mean([ps[i].argmax() == k for i, (_, k) in enumerate(gel) if k == s]) for s in sorted({k for _, k in gel})]
            puan = float(np.mean(dogru))
        else:
            puan = makro
        print(f"== tur {t}: XED doğrulama makro-F1 {makro:.4f} · geliştirme stil (sınıf dengeli) %{100 * puan:.1f}", flush=True)
        if puan > en_iyi:
            en_iyi = puan
            hedef.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), hedef / "model.pt")
            print("   kaydedildi", flush=True)

    # ONNX: iki çıkış; eşdeğerlik denetimi
    model.load_state_dict(torch.load(hedef / "model.pt"))
    model.eval()
    onnx_dir = KOK / "model" / "onnx"
    yeni = hedef / "duygu.onnx"
    adlar = ["input_ids", "attention_mask", "token_type_ids"]
    ornek_g = tok(["Bu bir deneme cümlesidir."], return_tensors="pt")
    torch.onnx.export(model, tuple(ornek_g[a] for a in adlar), str(yeni), input_names=adlar,
                      output_names=["logits", "stil_logits"], opset_version=17, dynamo=False,
                      dynamic_axes={**{a: {0: "parti", 1: "uzunluk"} for a in adlar},
                                    "logits": {0: "parti"}, "stil_logits": {0: "parti"}})
    import onnxruntime as ort
    o = ort.InferenceSession(str(yeni), providers=["CPUExecutionProvider"])
    g = tok(xd[:200], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="np")
    a, b = o.run(None, {k: g[k].astype(np.int64) for k in adlar})
    pt_d, pt_s = tahmin(model, tok, xd[:200], parti=200)
    sapma = max(float(np.abs(1 / (1 + np.exp(-a)) - pt_d).max()),
                float(np.abs(np.exp(b) / np.exp(b).sum(1, keepdims=True) - pt_s).max()))
    print(f"ONNX ↔ PyTorch en büyük sapma {sapma:.2e}")
    if sapma > 1e-3:
        print("KAPI GEÇİLMEDİ — model/onnx değişmedi")
        return 1
    if (onnx_dir / "duygu.onnx").exists():
        shutil.copy(onnx_dir / "duygu.onnx", onnx_dir / "duygu_tek_cikis.onnx.yedek")
    shutil.copy(yeni, onnx_dir / "duygu.onnx")
    a_ = json.loads((onnx_dir / "esikler.json").read_text(encoding="utf-8"))
    a_["stiller"] = list(STILLER)
    a_["stil_gelistirme_dengeli"] = round(en_iyi, 4)
    (onnx_dir / "esikler.json").write_text(json.dumps(a_, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazıldı: {onnx_dir / 'duygu.onnx'} (iki çıkış)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
