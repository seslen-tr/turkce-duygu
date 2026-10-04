# -*- coding: utf-8 -*-
"""
ONNX AKTARIMI — model/hf → model/onnx/duygu.onnx + tokenizer.json + esikler.json.

Adaylar küçükten büyüğe denenir, kapıdan geçen İLK aday yazılır:
  int8-kanal   dinamik int8, kanal başına ölçek, yalnız MatMul (~110 MB)
  fp32         sıkıştırmasız (~440 MB)
(04.10: düz dinamik int8 kapıdan geçmedi: sapma 0,38, makro-F1 −0,026.)
Kapı (doğrulama + nötr doğrulama kümeleri): olasılık sapması ≤ 0,05; en güçlü duygu %98 cümlede aynı;
"duygulu mu / nötr mü" kararı %98 cümlede aynı; makro-F1 farkı < 0,01.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import f1, notr_oku, oku, tahmin  # noqa: E402


def main() -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from turkce_duygu.model import DuyguModeli

    kaynak, hedef = KOK / "model" / "hf", KOK / "model" / "onnx"
    gecici = KOK / "model" / "gecici"
    gecici.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(kaynak)
    model = AutoModelForSequenceClassification.from_pretrained(kaynak).eval()
    ornek = tok(["Bu bir deneme cümlesidir."], return_tensors="pt")
    adlar = ["input_ids", "attention_mask", "token_type_ids"]
    dinamik = {a: {0: "parti", 1: "uzunluk"} for a in adlar} | {"logits": {0: "parti"}}
    torch.onnx.export(model, tuple(ornek[a] for a in adlar), str(gecici / "fp32.onnx"), input_names=adlar,
                      output_names=["logits"], dynamic_axes=dinamik, opset_version=17, dynamo=False)
    tok.backend_tokenizer.save(str(gecici / "tokenizer.json"))
    shutil.copy(kaynak / "esikler.json", gecici / "esikler.json")
    ayar = json.loads((kaynak / "esikler.json").read_text(encoding="utf-8"))
    esik, notr_esik = np.array(ayar["esik"]), ayar["notr_esik"]

    xd, yd = oku("dogrulama")
    xn = notr_oku("dogrulama")
    p_pt = tahmin(model, tok, xd)
    pn_pt = tahmin(model, tok, xn)

    def duygulu(p):
        return ((p / esik).max(1) / notr_esik) >= 1

    adaylar = [("int8-kanal", lambda y: quantize_dynamic(str(gecici / "fp32.onnx"), str(y), weight_type=QuantType.QInt8,
                                                          per_channel=True, op_types_to_quantize=["MatMul"])),
               ("fp32", lambda y: shutil.copy(gecici / "fp32.onnx", y))]
    for ad, yap in adaylar:
        yap(gecici / "duygu.onnx")
        dm = DuyguModeli(gecici)
        t0 = time.perf_counter()
        p_ox = np.concatenate([dm.olasiliklar(xd[i:i + 64]) for i in range(0, len(xd), 64)])
        sure = time.perf_counter() - t0
        pn_ox = np.concatenate([dm.olasiliklar(xn[i:i + 64]) for i in range(0, len(xn), 64)])
        tek = []
        for c in xd[:50]:
            t = time.perf_counter()
            dm.olasiliklar([c])
            tek.append(time.perf_counter() - t)
        sapma = float(max(np.abs(p_pt - p_ox).max(), np.abs(pn_pt - pn_ox).max()))
        ayni = float(np.mean(p_pt.argmax(1) == p_ox.argmax(1)))
        karar = float(np.mean(np.concatenate([duygulu(p_pt) == duygulu(p_ox), duygulu(pn_pt) == duygulu(pn_ox)])))
        f_pt, f_ox = float(f1(yd, p_pt, esik).mean()), float(f1(yd, p_ox, esik).mean())
        boyut = (gecici / "duygu.onnx").stat().st_size / 1e6
        print(f"[{ad}] sapma {sapma:.4f} · aynı en güçlü duygu %{100 * ayni:.1f} · aynı nötr kararı %{100 * karar:.1f} · "
              f"makro-F1 PyTorch {f_pt:.4f} / ONNX {f_ox:.4f} · {boyut:.0f} MB · {1000 * sure / len(xd):.1f} ms/cümle "
              f"(toplu), {1000 * float(np.median(tek)):.1f} ms (tek cümle)", flush=True)
        del dm
        if sapma <= 0.05 and ayni >= 0.98 and karar >= 0.98 and abs(f_pt - f_ox) < 0.01:
            hedef.mkdir(parents=True, exist_ok=True)
            for f in ("duygu.onnx", "tokenizer.json", "esikler.json"):
                shutil.copy(gecici / f, hedef / f)
            shutil.rmtree(gecici, ignore_errors=True)
            print(f"yazıldı: {hedef} ({ad})")
            return 0
        print(f"[{ad}] kapıdan geçmedi")
    print("hiçbir aday kapıdan geçmedi — model/onnx yazılmadı")
    return 1


if __name__ == "__main__":
    sys.exit(main())
