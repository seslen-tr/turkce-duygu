from pathlib import Path

from turkce_duygu.eylem import eylem, fiil_kipi

ORNEK = [s.split("\t") for s in (Path(__file__).parent / "eylem_literatur.tsv").read_text(encoding="utf-8").splitlines()
         if s and not s.startswith("#")]


def test_literatur():
    yanlis = [(c, b, eylem(c).tur, k) for c, b, k in ORNEK if b != "belirsiz" and eylem(c).tur != b]
    assert not yanlis, "\n".join(f"{c} → {t} (beklenen {b}; {k})" for c, b, t, k in yanlis)


def test_fiil_kipleri():
    assert fiil_kipi("gidin") == "emir"          # git → gid-in
    assert fiil_kipi("edin") == "emir"           # et → ed-in
    assert fiil_kipi("yavaşlayın") == "emir"
    assert fiil_kipi("dokunmayın") == "yasak"
    assert fiil_kipi("gidelim") == "oneri"
    assert fiil_kipi("kitap") == ""


def test_ipucu():
    e = eylem("Bana biraz yardım edebilir misiniz?")
    assert e.ipucu == ["edebilir", "misiniz"]
    assert eylem("Eyvah, anahtarı evde unuttum.").unlem == "uzuntu"
