# -*- coding: utf-8 -*-
"""
HUGGING FACE'E YÜKLE — modeli ve bu klasördeki model kartını (README.md) Hugging Face deposuna koyar.

Önce: pip install huggingface_hub && huggingface-cli login   (yazma yetkili kendi anahtarınız; depoda/sohbette değil)
    python hf/yukle.py            gizli depo (önce kendiniz bakın)
    python hf/yukle.py --acik     herkese açık
Depo adı TURKCE_DUYGU_HF ortam değişkeniyle değiştirilebilir (paket de aynı değişkeni okur).
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
DEPO = os.environ.get("TURKCE_DUYGU_HF", "seslen-tr/turkce-duygu")
DOSYALAR = {'model/onnx/duygu.onnx': 'duygu.onnx', 'model/onnx/tokenizer.json': 'tokenizer.json', 'model/onnx/esikler.json': 'esikler.json'}            # yerel yol (proje köküne göre) → depodaki yol


def main() -> None:
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument("--acik", action="store_true", help="depoyu herkese açık yap (varsayılan gizli)")
    a = ap.parse_args()
    eksik = [y for y in DOSYALAR if not (KOK / y).exists()]
    if eksik:
        raise SystemExit("eksik model dosyası: " + ", ".join(eksik))
    with tempfile.TemporaryDirectory() as t:
        for yerel, hedef in DOSYALAR.items():
            (Path(t) / hedef).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(KOK / yerel, Path(t) / hedef)
        shutil.copy(Path(__file__).with_name("README.md"), Path(t) / "README.md")
        api = HfApi()
        api.create_repo(DEPO, private=not a.acik, exist_ok=True)
        api.upload_folder(folder_path=t, repo_id=DEPO, commit_message="model ve model kartı")
    print(f"yüklendi: https://huggingface.co/{DEPO} ({'açık' if a.acik else 'gizli'})")


if __name__ == "__main__":
    main()
