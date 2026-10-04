# -*- coding: utf-8 -*-
"""
ÖLÇÜM — sınav kümesinde (eğitim ve eşik seçiminde hiç kullanılmadı):

  1) XED Türkçe sınav: makro-F1 (duygu başına doğrulamada seçilen eşikle), ilk tahmin doğruluğu
     (en olası duygu altın etiketlerden biri mi). Karşılaştırma: sınıf sıklığı tabanı, TF-IDF +
     lojistik regresyon (karakter 2-5 gram), ince ayarlı BERTurk (ONNX int8).
  2) Nötr: Nötr sınav cümlelerinin (Vikipedi + Common Voice) yüzde kaçı "duygu yok" (nötr) çıkıyor; XED sınavındakilerin
     yüzde kaçı duygulu çıkıyor.

    python degerlendirme/olc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import esik_sec, f1, oku  # noqa: E402
from egit.veri import DUYGULAR  # noqa: E402


def ozet(ad: str, y: np.ndarray, p: np.ndarray, esik: np.ndarray) -> dict:
    f = f1(y, p, esik)
    ilk = float(np.mean(y[np.arange(len(y)), p.argmax(1)] > 0))
    print(f"{ad:28s} makro-F1 {f.mean():.3f}  ilk tahmin %{100 * ilk:.1f}  "
          + " ".join(f"{d[:5]} {v:.2f}" for d, v in zip(DUYGULAR, f)))
    return {"makro_f1": round(float(f.mean()), 4), "ilk_dogru": round(ilk, 4),
            "duygu_f1": {d: round(float(v), 4) for d, v in zip(DUYGULAR, f)}}


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    xe, ye = oku("egitim")
    xd, yd = oku("dogrulama")
    xs, ys = oku("sinav")
    sonuc = {"sinav_cumle": len(xs)}

    # Sınıf sıklığı tabanı: her cümleye eğitimdeki sıklıklar
    p0 = np.tile(ye.mean(0), (len(xs), 1)) + np.random.default_rng(0).normal(0, 1e-6, (len(xs), len(DUYGULAR)))
    sonuc["siklik"] = ozet("sıklık tabanı", ys, p0, np.full(len(DUYGULAR), 0.0))

    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xe, Xd, Xs = v.fit_transform(xe), v.transform(xd), v.transform(xs)
    pd_, ps_ = np.zeros_like(yd), np.zeros_like(ys)
    for j in range(len(DUYGULAR)):
        lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xe, ye[:, j])
        pd_[:, j], ps_[:, j] = lr.predict_proba(Xd)[:, 1], lr.predict_proba(Xs)[:, 1]
    sonuc["tfidf_lr"] = ozet("TF-IDF + lojistik regresyon", ys, ps_, esik_sec(yd, pd_))

    from turkce_duygu.model import DuyguModeli
    dm = DuyguModeli()
    p = np.concatenate([dm.olasiliklar(xs[i:i + 64]) for i in range(0, len(xs), 64)])
    sonuc["berturk_onnx_int8"] = ozet("BERTurk ince ayar (ONNX)", ys, p, dm.esik)

    notr = [s for s in (KOK / "veri" / "notr_sinav.txt").read_text(encoding="utf-8").splitlines() if s]
    pn = np.concatenate([dm.olasiliklar(notr[i:i + 64]) for i in range(0, len(notr), 64)])
    notr_oran = float(np.mean(dm.guc(pn).max(1) < 1))
    duygulu = float(np.mean(dm.guc(p).max(1) >= 1))
    print(f"Nötr sınav cümlelerinin (Vikipedi + Common Voice) %{100 * notr_oran:.1f}'i nötr; XED sınav cümlelerinin %{100 * duygulu:.1f}'i duygulu")
    sonuc["vikipedi_notr"] = round(notr_oran, 4)
    sonuc["xed_duygulu"] = round(duygulu, 4)
    (KOK / "degerlendirme" / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
