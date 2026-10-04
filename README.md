# turkce-duygu

Türkçe metinden **seslendirme stili** seçer: her cümle için nötr, neşeli, heyecan, uyarı, merak
ya da özür tonunu ve bir yoğunluk değerini (0–1) verir. GPU ve internet gerekmez.

```python
from turkce_duygu import metin_cozumle

for r in metin_cozumle("Günaydın! Bugün hava harika. Şemsiyeni unutma, öğleden sonra yağmur yağabilir. "
                       "Bana bir iyilik yapabilir misin?"):
    print(r.stil, r.yogunluk, r.eylem, r.cumle)
```

```
python -m turkce_duygu "Eyvah, anahtarı evde unuttum. Kapıyı açabilir misiniz?"
```

## Neden

Duygulu TTS modelleri (StyleTTS2, duygu etiketli VITS vb.) hangi cümlenin hangi tonla okunacağını
bilmek ister. Türkçe için bunu metinden çıkaran açık bir araç yoktu. Genel duygu sınıflandırıcıları
iki şeyi kaçırır:

- **Biçim:** "Bana yardım edebilir misiniz?" soru işaretlidir ama merak değil ricadır.
  "Dikkat, zemin ıslak." duygusuz görünür ama uyarıdır.
- **Duygu:** "Bugün hava harika." biçimce düz bir bildirimdir ama neşeyle okunmalıdır.

Bu paket ikisini birleştirir: biçimi dilbilgisi kurallarıyla, duyguyu modelle çıkarır.

## Nasıl çalışır

**1. Konuşma eylemi (kurallar, `eylem.py`).** Kurallar kaynaklıdır. Her kural, kitaptaki numaralı
örneklerle sınanır ([tests/eylem_literatur.tsv](tests/eylem_literatur.tsv)).

| Eylem | Biçim | Örnek | Kaynak |
|---|---|---|---|
| soru | soru eki mI, ne-sözcüğü + ?, acaba | Kitabı Ahmet'e mi verdin? | G&K §19.1–19.2, §19.5.2 |
| rica / teklif | 2. kişi geniş zaman sorusu, -(y)Abilir mI | Bana biraz yardım edebilir misiniz? · Çay içer misiniz? | G&K §21.4.4.5 |
| danışma | 1. kişi istek + mI, -sAm/-sAk + acaba | Sana yardım edeyim mi? · Kime danışsam acaba? | G&K §21.4.4.3–4 |
| emir | emir kipi, -(y)In(Iz), -sIn, -sAnA, -mAlI | Bu parayı babana ver. · Lütfen kemerlerinizi bağlayınız. | G&K §21.4.4.2, §21.4.2.2 |
| yasak | olumsuz emir, -mAmAlI | Kabloya dokunmayın. | G&K §21.4.4.2, §21.4.2.2 |
| uyarı | "dikkat" ünlemi; sakın/aman + buyruk; buyruk + tehlike ya da olası kötü sonuç | Dikkat! Zemin ıslak. · Çöpe atma, yangın çıkabilir. | TDK; G&K §21.4.2.1 |
| özür | açık özür sözü; pişmanlık, sorumluluk üstlenme, telafi | Kusura bakmayın, toplantı uzadı. | Hatipoğlu 2003; Olshtain & Cohen 1983 |
| dilek | keşke, 1. kişi -sA | Keşke daha çok param olsa. | G&K §21.4.4.1 |
| öneri | -(y)AlIm, -(y)AyIm | Biraz konuşalım. | G&K §21.4.4.3 |
| ünlem | TDK'nın duygu tanımıyla ünlemler (yaşasın: sevinç, eyvah: üzüntü, vay: şaşma…) | Eyvah, anahtarı evde unuttum. | TDK; G&K §7.1.5 |

Emir ve istek biçimleri 1.806 eylem gövdesinden üretilir. Türetilmiş gövdeler (yanaş-tır-mayın,
kullan-ıl-masın) ve ünsüz yumuşaması (git → gidin) de tanınır. "Affedersiniz, istasyon ne tarafta?"
özür değil, dikkat çekme sözüdür. "Dikkatini çekti" uyarı değildir.

**2. Duygu (model, `model.py`).** BERTurk ([dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)),
[XED](https://github.com/Helsinki-NLP/XED) Türkçe verisiyle Plutchik'in 8 temel duygusuna ince
ayarlandı: öfke, beklenti, tiksinti, korku, sevinç, üzüntü, şaşkınlık, güven. ONNX int8 olarak
işlemcide çalışır.

**3. Stil (`stil.py`).** Önce biçim karar verir: soru → merak, özür → özür, uyarı/yasak → uyarı,
emir → hafif uyarı, rica → nötr. Biçimin söylemediği yerde duygu: sevinç/güven → neşeli,
şaşkınlık/beklenti → heyecan, korku/öfke/tiksinti → uyarı, üzüntü → özür. Hiçbir duygu eşiği
geçmezse nötr. Ünlem işareti ve pekiştirme sözcükleri (çok, mutlaka, kesinlikle, hemen) yoğunluğu
artırır. Paragrafta, duygulu bir cümlenin ardındaki eşiğe yakın cümle tonu yarı yoğunlukla sürdürür.
Eşleme tablo hâlinde durur, değiştirilebilir.

## Doğruluk

**Duygu modeli.** XED Türkçe sınav kümesi, 904 cümle. Eğitimde ve eşik seçiminde kullanılmadı.

| Model | Makro-F1 | İlk tahmin doğru |
|---|---|---|
| Sınıf sıklığı tabanı | 0,225 | %20,2 |
| TF-IDF + lojistik regresyon | 0,289 | %28,8 |
| **BERTurk ince ayar (ONNX)** | **0,327** | **%35,8** |

- "İlk tahmin doğru": en olası duygu altın etiketlerden biri.
- Doğrulamada makro-F1 0,390'dı; aradaki fark, eşiklerin doğrulama kümesine uymasından gelir.
- **Nötr ayrımı:** sınav kümesindeki nötr cümlelerin (Vikipedi + Common Voice) %95,1'i nötr, XED
  cümlelerinin %98,2'si duygulu bulunur. Nötr ve duygulu örnekler farklı kaynaklardan geldiği için bu
  oranın bir kısmı üslup farkından gelir; üslubu ayrı ölçmek için [turkce-uslup](../turkce-uslup).
- **ONNX:** int8 sıkıştırma kalite kapısını geçmedi (makro-F1 −0,026), model fp32 yayımlanır (443 MB).
  Tek cümle 13 ms, PyTorch ile birebir aynı.

**Seslendirme stili (kurallar + model).** Seslen ses atölyesinin ton havuzlarında ölçüldü: kurallar
yalnız ilk yarılara bakılarak geliştirildi, sayılar hiç incelenmeyen ikinci yarılardan. Havuzlar
yayımlanmıyor; betik `degerlendirme/stil_havuz.py`.

| Ton | Doğru | En sık karıştığı |
|---|---|---|
| uyarı | %92,6 | kızgın |
| merak | %88,9 | heyecan |
| özür | %84,6 | üzgün |
| nötr | %84,3 | heyecan |
| heyecan | %43,4 | neşeli |
| kızgın | %37,7 | merak (öfkeli soru), uyarı (öfkeli emir) |
| üzgün | %33,7 | nötr |
| neşeli | %29,7 | nötr |

Biçimle taşınan tonlar (soru, uyarı, özür) güvenilirdir. Yalnız duyguyla taşınanlar (neşe, üzüntü,
öfke) zayıftır; altyazı verisiyle eğitilen model yazılı cümlelerdeki sakin duyguyu çoğu zaman nötr
bulur. Yanılgılar çoğunlukla komşu tona (neşeli ↔ heyecan) ya da nötre düşer. Seslendirmede bu,
yanlış duygudan daha zararsızdır. Biçimi ezen güçlü duygu kuralı (`baskin`) ve duyarlılık ayarı
denendi (`degerlendirme/stil_ayar.py`). Geliştirme yarısında seçilen ayar sınavda toplamı yalnız %63,8'den
%64,1'e çıkardı, nötr doğruluğunu %84'ten %72'ye düşürdü; varsayılan olarak kapalıdır.

## Kurulum

```
pip install turkce-duygu[model]
```

Model dosyaları (duygu.onnx, tokenizer.json, esikler.json; ~110 MB) Hugging Face'te yayımlanır.
`TURKCE_DUYGU_MODEL` ortam değişkeniyle klasör gösterilir. Model yoksa paket yalnız kurallarla
çalışır.

**Yeniden eğitmek** (GPU gerekmez, işlemcide yaklaşık bir saat):
```
pip install turkce-duygu[egitim]
python egit/veri.py        # XED (sabit sürüm) + Vikipedi nötr örnekleri, MD5'e göre bölme
python egit/egit.py        # BERTurk ince ayar; en iyi tur doğrulama makro-F1'ine göre
python egit/aktar.py       # ONNX int8 + eşdeğerlik kapısı
python degerlendirme/olc.py
```

## Sınırlar

- XED Türkçe etiketleri İngilizce/Fince insan etiketlerinin altyazılara **aktarımıdır**. Etiketler
  gürültülüdür. İnsan etiketçiler arasında bile uyum düşüktür (XED'in kendi İngilizce makro-F1'i ~0.5).
  Sayılar bu tavana göre okunmalıdır.
- Altyazı dili konuşma dilidir. Haber ve ansiklopedi metninde model çoğunlukla nötr kalır.
  Bu, seslendirme için istenen davranıştır.
- Konuşma eylemi biçimden çıkarılır. Tonlamayla taşınan ironi ve söz sanatı soruları bağlamsız
  çözülemez. Bunlar sınav dosyasında "belirsiz" olarak işaretlidir.

## Kaynaklar

- Göksel, A. & Kerslake, C. (2005). *Turkish: A Comprehensive Grammar.* Routledge.
- Hatipoğlu, Ç. (2003). Turkish apologies.
- Olshtain, E. & Cohen, A. D. (1983). Apology: a speech act set. *Sociolinguistics and Language Acquisition*, 18–35.
- Öhman, E., Pàmies, M., Kajava, K. & Tiedemann, J. (2020). XED: A Multilingual Dataset for Sentiment
  Analysis and Emotion Detection. *COLING 2020*.
- Schweter, S. (2020). BERTurk – BERT models for Turkish. Zenodo.
- TDK Güncel Türkçe Sözlük (sozluk.gov.tr).

## Lisans

Kod MIT ([LICENSE](LICENSE)). Model ağırlıkları XED'den türediği için **CC BY 4.0** (atıf:
Öhman vd. 2020). `turkce_duygu/veri/fiiller.txt` Vikisözlük türevidir (CC BY-SA 4.0).

Kardeş projeler: [turkce-vurgu](../turkce-vurgu) (vurgulu okunuş), [turkce-normalize](../turkce-normalize)
(metin hazırlama), [turkce-plbert](../turkce-plbert), [turkce-tts-sinav](../turkce-tts-sinav).
