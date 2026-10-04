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
