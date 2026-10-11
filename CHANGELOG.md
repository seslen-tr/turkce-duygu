# Değişiklikler

## 0.1.1 — belgeler

- README ve CHANGELOG koda göre güncellendi (kurulum, kullanım, API, düzen); kod davranışı değişmedi.

## 0.1.0 — ilk açık sürüm

- İlk yayın. Ayrıntılı ölçümler README'de.
- Kurallı konuşma eylemi (`eylem.py`): soru, rica/teklif, danışma, emir, yasak, uyarı, özür, dilek, öneri,
  ünlem; kurallar kaynaklardaki numaralı örneklerle sınanır (`tests/eylem_literatur.tsv`).
- İki çıkışlı BERTurk modeli (ONNX, fp32): 8 Plutchik duygusu (XED Türkçe) ve 8 seslendirme stili
  (açık stil seti, 908 cümle, CC0).
- Stil seçimi (`stil.py`): önce biçim, sonra modelin stil çıkışı, yoksa duygu eşlemesi; yoğunluk 0–1,
  fısıltı yalnız açık sözle, sesin olmayan stili için yedek stil, paragraf içi süreklilik.
- Model yerelde yoksa Hugging Face'ten (`seslen-tr/turkce-duygu`) iner; `[model]` eki olmadan yalnız
  kurallarla çalışır. `turkce-normalize` kuruluysa onun cümle bölücüsü kullanılır.
- Komut satırı: `python -m turkce_duygu "metin"`.
