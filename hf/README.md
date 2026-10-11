---
language: tr
license: cc-by-4.0
library_name: onnx
pipeline_tag: text-classification
base_model: dbmdz/bert-base-turkish-cased
tags: [turkish, emotion, speaking-style, tts, onnx]
---

# turkce-duygu — Türkçe seslendirme stili modeli

> **English:** Fine-tuned BERTurk (ONNX, CPU) with two heads: Plutchik's 8 emotions (trained on Turkish XED)
> and 8 TTS speaking styles (neutral, cheerful, excited, warning, curious, apologetic, sad, angry). Used by the
> turkce-duygu package together with rule-based speech-act detection.

İnce ayarlı BERTurk, iki çıkışlı, ONNX (fp32, ~445 MB; int8 kalite kapısını geçmedi). İşlemcide çalışır.

| Dosya | İçerik |
|---|---|
| `duygu.onnx` | girdiler `input_ids`, `attention_mask`, `token_type_ids` (paket 64 parçayla keser); çıkışlar `logits` (8 duygu), `stil_logits` (8 stil) |
| `tokenizer.json` | BERTurk sözcük parçalayıcı |
| `esikler.json` | duygu başına karar eşikleri (doğrulama kümesinde ayarlandı) |

## Kullanım

```python
# pip install "turkce-duygu[model]"      → model ilk kullanımda buradan iner
from turkce_duygu import metin_cozumle
for r in metin_cozumle("Eyvah, anahtarı evde unuttum. Kapıyı açabilir misiniz?"):
    print(r.stil, r.yogunluk, r.cumle)
```

## Ölçüler (ayrılmış sınav kümesi)

XED Türkçe sınav kümesi, 904 cümle; eğitimde ve eşik seçiminde kullanılmadı.

| | Makro-F1 (8 duygu) | İlk tahmin doğru |
|---|---|---|
| Sınıf sıklığı tabanı | 0,225 | %20,2 |
| TF-IDF + lojistik regresyon | 0,289 | %28,8 |
| **Bu model** | **0,327** | **%35,8** |

Nötr ayrımı: nötr sınav cümlelerinin (Vikipedi + Common Voice) %95,1'i nötr, XED sınav cümlelerinin %98,2'si
duygulu bulunur (kaynak farkı bu oranı kısmen şişirir).

Seslendirme stili (kurallar + bu model, 8 ton): toplam **%85,6** (yalnız duygu eşlemesiyle %63,8).
Ayrıntı ve sınırlılıklar: paketin README'si.

## Veri ve lisans

- Duygu çıkışı: [XED](https://github.com/Helsinki-NLP/XED) Türkçe (CC BY 4.0; Öhman vd. 2020).
- Stil çıkışı: projeye özel stil seti (908 cümle, CC0, bir dil modeli yardımıyla yazıldı), nötr örnekler ve
  düşük ağırlıkla duygudan eşlenmiş XED cümleleri.
- Başlangıç: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) (MIT).
- Ağırlıklar **CC BY 4.0** (XED türevi). Kod: MIT.

## Sınırlılıklar

Altyazı verisiyle eğitilen duygu çıkışı yazılı metindeki sakin duyguyu çoğu zaman nötr bulur; stil çıkışı bu
açığı kapatır ama stil seti kısa örnek cümlelerden oluşur, gerçek metinde başarı daha düşük olabilir. Yanılgılar
çoğunlukla komşu tona (neşeli ↔ heyecan, kızgın ↔ uyarı) ya da nötre düşer.
