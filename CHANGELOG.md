# Değişiklikler

## 0.1.2 — kısa yanıt, değilleme, düz emir

- Ünlemsiz kısa yanıt nötr: "Tamam.", "Evet.", "Peki.", "Hayır.", "Teşekkür ederim." (model "Tamam."ı neşeli 0,85,
  "Evet."i kızgın 0,85 okutuyordu); ünlemliyse hafif heyecan ("Evet!").
- Değillenen olumsuz duygu nötr: "Çok üzgün değilim.", "Kötü değildi." (önceden üzgün 0,80 / neşeli). Olumlu
  duygunun değillenmesi değişmedi ("Hiç mutlu değilim." üzgün).
- Düz emir nötr: tarif ve kılavuz ("Önce fırını ısıtın.", "Ayarlar menüsünü açın.") uyarı oluyordu. Ünlem,
  pekiştirme ("hemen", "sakın"), tehlike sözü ("dikkatle", "yakarsın", "uzak durun") varsa ya da model uyarı
  görüyorsa uyarı kalır. Havuz sınavı (2209 cümle): %85,6 aynı.

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
