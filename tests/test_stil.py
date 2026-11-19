from turkce_duygu import Cozumleyici, cumlelere_bol

KURAL = Cozumleyici(model=False)          # model olmadan: yalnız eylem kuralları


def stil(c):
    r = KURAL.cozumle(c)
    return r.stil, r.yogunluk


def test_bicimden_stil():
    assert stil("Bu parselin sınırı neden eğri çizilmiş?")[0] == "merak"
    assert stil("Bana biraz yardım edebilir misiniz?")[0] == "notr"       # rica: merak değil
    assert stil("Özür dilerim, geç kaldım.")[0] == "ozur"
    assert stil("Dikkat, zemin ıslak.")[0] == "uyari"
    assert stil("Yaşasın, tatil başladı!")[0] == "neseli"
    assert stil("Eyvah, anahtarı evde unuttum.")[0] == "uzgun"
    eski = Cozumleyici(model=False, stiller=("notr", "neseli", "heyecan", "uyari", "merak", "ozur"))
    assert eski.cozumle("Eyvah, anahtarı evde unuttum.").stil == "ozur"      # üzgün yoksa yedek
    assert stil("Kulağına eğilip yavaşça fısıldadı.")[0] == "fisilti"
    assert stil("Toplantı yarın saat onda başlayacak.") == ("notr", 0.0)


def test_pekistirme():
    _, hafif = stil("Kabloya dokunmayın.")
    _, guclu = stil("Kabloya kesinlikle dokunmayın!")
    assert guclu > hafif


def test_cumle_bolme():
    assert cumlelere_bol("Dr. Ayşe geldi. Saat 3.30'da başladık! Sonra? Bitti.") == [
        "Dr. Ayşe geldi.", "Saat 3.30'da başladık!", "Sonra?", "Bitti."]


def test_kisa_yanit_degilleme_duz_emir():
    # 18.11.2026: model "Tamam." neşeli, "Evet." kızgın, "Çok üzgün değilim." üzgün okutuyordu; tarif uyarı oluyordu
    for c in ("Tamam.", "Evet.", "Peki.", "Teşekkür ederim.", "Çok üzgün değilim.", "Kötü değildi.",
              "Önce fırını ısıtın.", "Ayarlar menüsünü açın."):
        assert stil(c) == ("notr", 0.0), c
    assert stil("Evet!")[0] == "heyecan"
    assert stil("Tepsiyi eldivensiz tutma, elini yakarsın.")[0] == "uyari"     # tehlike sözü: uyarı kalır
    assert stil("Hemen buraya gel!")[0] == "uyari"                             # ünlem + pekiştirme


def test_kovma_sessizlik_yalvarma():
    # 19.11.2026: "Bırakın beni!" neşeli, "Sessiz ol, bebek uyuyor." kızgın, "Lütfen gitme." uyarı okunuyordu
    for c in ("Bırakın beni!", "Rahat bırak beni.", "Defol git buradan!"):
        assert stil(c)[0] == "kizgin", c
    assert stil("Beni rahat bırakın lütfen!")[0] == "uzgun"
    for c in ("Sessiz ol, bebek uyuyor.", "Yavaş konuş, herkes uyuyor.", "Şşş, sessiz olun."):
        assert stil(c)[0] == "fisilti", c
    for c in ("Lütfen gitme.", "Lütfen beni bırakma.", "Lütfen bana kızma."):
        assert stil(c)[0] == "uzgun", c
    assert stil("Lütfen kapıyı kapatmayı unutma.") == ("notr", 0.0)
    assert stil("Lütfen yere çöp atmayın, ceza uygulanır.")[0] == "uyari"       # tehlike sözü


def test_kazan_kaza_degil_var_emir_degil():
    from turkce_duygu import eylem
    assert eylem("Yaşasın, kazandık!").tur != "uyari"
    assert stil("Yaşasın, kazandık!")[0] == "neseli"
    assert eylem("Korkuyorum, karanlıkta biri var.").tur != "emir"
    assert eylem("Kaza yapmamak için dikkatli sürün.").tur == "uyari"
    assert stil("Bunun üç yararı var:") == ("notr", 0.0)                      # liste girişi
