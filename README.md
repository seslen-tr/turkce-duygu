# turkce-duygu

> **English:** Picks a speaking style for each Turkish sentence for expressive TTS: neutral, cheerful, excited, warning, curious, apologetic, sad, angry or whisper, with an intensity (0–1). Combines rule-based speech-act detection (question, request, warning, apology) with a fine-tuned BERTurk emotion model (ONNX, CPU only; downloaded from Hugging Face on first use).

Türkçe metinden **seslendirme stili** seçer: her cümle için nötr, neşeli, heyecan, uyarı, merak,
özür, üzgün, kızgın ya da fısıltı tonunu ve bir yoğunluk değerini (0–1) verir. GPU ve internet gerekmez.

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

**2. Model (`model.py`).** BERTurk ([dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased)),
iki çıkışlı. ONNX olarak işlemcide çalışır.

- **Duygu çıkışı:** [XED](https://github.com/Helsinki-NLP/XED) Türkçe verisiyle Plutchik'in 8 temel
  duygusu (öfke, beklenti, tiksinti, korku, sevinç, üzüntü, şaşkınlık, güven).
- **Stil çıkışı** (`egit/egit_stil.py`): 8 seslendirme stilinin olasılığı. Bu proje için yazılmış açık
  stil setiyle (`turkce_duygu/veri/stil_seti`, 908 cümle, CC0; sentetik: bir dil modeli yardımıyla
  yazıldı), nötr örneklerle ve duygudan eşlenmiş
  XED cümleleriyle (düşük ağırlıkla) eğitildi.

**3. Stil (`stil.py`).** Önce biçim karar verir: soru → merak, özür → özür, uyarı/yasak → uyarı,
emir → hafif uyarı, rica → nötr. Biçimin söylemediği yerde modelin stil çıkışı karar verir. Stil
çıkışı yoksa (eski tek çıkışlı model) duygu eşlemesi kullanılır: sevinç/güven → neşeli,
şaşkınlık/beklenti → heyecan, korku → uyarı, öfke/tiksinti → kızgın, üzüntü → üzgün. Hiçbir duygu
eşiği geçmezse nötr. Çok güçlü öfke ya da üzüntü biçimi ezer: "Kaç kere söyleyeceğim sana, kapıyı
kapat!" emir biçimindedir ama kızgın okunur. Fısıltı metinden seçilmez; yalnız "fısıldadı" gibi açık bir söz geçerse önerilir. Sesin bir
stili yoksa en yakınına düşülür (üzgün → özür, kızgın → uyarı). Ünlem işareti ve pekiştirme sözcükleri (çok, mutlaka, kesinlikle, hemen) yoğunluğu
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

**Seslendirme stili (kurallar + model).** Seslen ses atölyesinin ton havuzlarında ölçüldü. Kurallar
ve eşikler yalnız ilk yarılara bakılarak seçildi; sayılar hiç incelenmeyen ikinci yarılardan (2.209
cümle). Havuzlar yayımlanmıyor; betik `degerlendirme/stil_havuz.py`.

| Ton | Stil çıkışıyla | Yalnız duygu çıkışıyla | En sık karıştığı (stil çıkışıyla) |
|---|---|---|---|
| uyarı | **%94,9** | %92,6 | kızgın |
| nötr | **%90,1** | %84,3 | üzgün, neşeli |
| özür | **%88,0** | %84,6 | üzgün |
| heyecan | **%87,7** | %43,4 | neşeli |
| merak | %84,0 | **%88,9** | kızgın (öfkeli soru) |
| neşeli | **%81,7** | %29,7 | nötr |
| üzgün | **%78,8** | %33,7 | nötr |
| kızgın | **%72,1** | %37,7 | merak, uyarı |
| **toplam** | **%85,6** | %63,8 | |

Biçimle taşınan tonlar (soru, uyarı, özür) iki yolda da güvenilirdir. Yalnız duyguyla taşınan tonlar
(neşe, heyecan, üzüntü, öfke) duygu çıkışıyla zayıf kalıyordu: altyazıyla eğitilen model yazılı
cümlelerdeki sakin duyguyu çoğu zaman nötr bulur. Stil çıkışı bu açığı kapatır. Yanılgılar çoğunlukla
komşu tona (neşeli ↔ heyecan, kızgın ↔ uyarı) ya da nötre düşer. Seslendirmede bu, yanlış duygudan
daha zararsızdır.

**Bu sayıları okurken:**

- Stil seti ile havuz cümleleri aynı türdendir: ikisi de seslendirme için yazılmış kısa örnek
  cümlelerdir. Bu yüzden sonuçlar gerçek kullanıcı metninde muhtemelen daha düşüktür.
- Üzgün ve kızgın havuzlarıyla birebir ya da çok benzer olan stil seti cümleleri eğitimden önce
  ayıklandı (`degerlendirme/sizinti_temizle.py`). Yine de bu iki tonun sınavı ötekiler kadar
  bağımsız değildir.
- Biçimi ezen duygu kuralı (`baskin`) duygu çıkışıyla varsayılan olarak kapalıdır. Altyazıyla
  eğitilen model düz soruların çoğunda da öfke görüyor: kural 1,5'te açıkken merak %89'dan %18'e
  düşüyor. Stil çıkışıyla aynı iş `stil_baskin` ile yapılır (varsayılan 0,6). Eşikler
  `Cozumleyici(ayar={...})` ile değiştirilebilir.

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
python egit/aktar.py       # ONNX + eşdeğerlik kapısı (int8 kapıyı geçmezse fp32)
python egit/egit_stil.py   # stil çıkışını ekler (iki çıkışlı ONNX)
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

Kod MIT ([LICENSE](LICENSE)). Stil seti (`turkce_duygu/veri/stil_seti`) CC0: cümleler bu proje için bir dil modeli
yardımıyla yazıldı (sentetik), gerçek kişilere ya da yayımlanmış metinlere dayanmaz. Model ağırlıkları XED'den türediği için **CC BY 4.0** (atıf:
Öhman vd. 2020). `turkce_duygu/veri/fiiller.txt` Vikisözlük türevidir (CC BY-SA 4.0).

Kardeş projeler: [turkce-vurgu](../turkce-vurgu) (vurgulu okunuş), [turkce-normalize](../turkce-normalize)
(metin hazırlama), [turkce-plbert](../turkce-plbert), [turkce-tts-sinav](../turkce-tts-sinav).
