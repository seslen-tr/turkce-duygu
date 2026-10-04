# -*- coding: utf-8 -*-
"""
DUYGU MODELİ — ince ayarlı BERTurk'ün ONNX (int8) sürümünü işlemcide çalıştırır.

İki çıkışlı modelde (egit/egit_stil.py) ikinci çıkış 8 seslendirme stilinin olasılıklarıdır.
Model klasörü (öncelik sırasıyla): DuyguModeli(klasor), TURKCE_DUYGU_MODEL ortam değişkeni,
paketin yanındaki model/, depo kökündeki model/onnx/. İçinde: duygu.onnx, tokenizer.json, esikler.json.
Gereken: onnxruntime, tokenizers (torch gerekmez).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

_PAKET = Path(__file__).resolve().parent


def hazirla(metin: str) -> str:
    """Model girdisi: altyazı tirelerini ve fazla boşluğu atar (eğitimde de aynısı uygulanır)."""
    return " ".join(re.sub(r"^[\s\-–—]+", "", metin).split())


def _klasor(klasor: str | Path | None) -> Path:
    adaylar = [Path(klasor)] if klasor else []
    if os.environ.get("TURKCE_DUYGU_MODEL"):
        adaylar.append(Path(os.environ["TURKCE_DUYGU_MODEL"]))
    adaylar += [_PAKET / "model", _PAKET.parent / "model" / "onnx"]
    for a in adaylar:
        if (a / "duygu.onnx").exists():
            return a
    raise FileNotFoundError("duygu modeli bulunamadı; TURKCE_DUYGU_MODEL ile klasörü gösterin "
                            "(duygu.onnx, tokenizer.json, esikler.json). Aranan: " + ", ".join(map(str, adaylar)))


class DuyguModeli:
    UZUNLUK = 64

    def __init__(self, klasor: str | Path | None = None, is_parcacigi: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        k = _klasor(klasor)
        ayar = json.loads((k / "esikler.json").read_text(encoding="utf-8"))
        self.duygular: list[str] = ayar["duygular"]
        self.esik = np.array(ayar["esik"], dtype=np.float32)
        self.notr_esik = float(ayar["notr_esik"])
        self.stil_ayar: dict = ayar.get("stil", {})          # degerlendirme/stil_ayar.py
        self.tok = Tokenizer.from_file(str(k / "tokenizer.json"))
        self.tok.enable_truncation(self.UZUNLUK)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]") or 0)
        se = ort.SessionOptions()
        if is_parcacigi:
            se.intra_op_num_threads = is_parcacigi
        self.oturum = ort.InferenceSession(str(k / "duygu.onnx"), se, providers=["CPUExecutionProvider"])
        self._girdiler = {g.name for g in self.oturum.get_inputs()}
        cikis = [o.name for o in self.oturum.get_outputs()]
        self.stiller: list[str] | None = ayar.get("stiller") if "stil_logits" in cikis else None

    def tahmin(self, cumleler: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        """(duygu olasılıkları (n, 8) sigmoid, stil olasılıkları (n, 8) softmax ya da None)."""
        if not cumleler:
            return np.zeros((0, len(self.duygular)), dtype=np.float32), None
        kod = self.tok.encode_batch([hazirla(c) for c in cumleler])
        g = {"input_ids": np.array([e.ids for e in kod], dtype=np.int64),
             "attention_mask": np.array([e.attention_mask for e in kod], dtype=np.int64),
             "token_type_ids": np.array([e.type_ids for e in kod], dtype=np.int64)}
        cikti = self.oturum.run(None, {k: v for k, v in g.items() if k in self._girdiler})
        d = 1 / (1 + np.exp(-cikti[0]))
        if not self.stiller:
            return d, None
        s = np.exp(cikti[1] - cikti[1].max(1, keepdims=True))
        return d, s / s.sum(1, keepdims=True)

    def olasiliklar(self, cumleler: list[str]) -> np.ndarray:
        """(n, 8) sigmoid olasılıkları, self.duygular sırasıyla."""
        return self.tahmin(cumleler)[0]

    def guc(self, p: np.ndarray) -> np.ndarray:
        """Olasılığın kendi eşiğine oranı, nötr eşiğiyle ölçeklenmiş: >1 ise duygu var sayılır."""
        return (p / self.esik) / self.notr_esik
