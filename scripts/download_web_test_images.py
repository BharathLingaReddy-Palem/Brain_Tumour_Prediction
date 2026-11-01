"""Download a deterministic labeled sample; never select images by model output."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time
from urllib.parse import quote

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "webtesting_folder"
HF_REPO = "Simezu/brain-tumour-MRI-scan"
GH_REPO = "SartajBhuvaji/Brain-Tumor-Classification-DataSet"


def get(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download(record):
    path = DEST / record["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    data = path.read_bytes() if path.exists() else get(record["download_url"]).content
    expected = record.pop("expected_sha256", None)
    if expected and hashlib.sha256(data).hexdigest() != expected:
        data = get(record["download_url"]).content
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source content changed for pinned image: {record['relative_path']}")
    with Image.open(BytesIO(data)) as image:
        image.load()
        rgb = image.convert("RGB")
        record["size"] = list(rgb.size)
        record["pixel_sha256"] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    path.write_bytes(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class-per-source", type=int, default=25)
    args = parser.parse_args()
    rng = random.Random(20261101)
    previous = json.loads((DEST/"manifest.json").read_text(encoding="utf-8")) if (DEST/"manifest.json").exists() else {}
    previous_images = {r["relative_path"]: r for r in previous.get("images", [])}
    revisions = {r["source"]: r["source_revision"] for r in previous.get("images", [])}
    hf_revision = revisions.get("hf_nickparvar") or get(f"https://huggingface.co/api/datasets/{HF_REPO}").json()["sha"]
    gh_ref = revisions.get("sartaj", "master")
    tree = get(f"https://api.github.com/repos/{GH_REPO}/git/trees/{gh_ref}?recursive=1").json()
    gh_revision = tree["sha"]
    records = []
    for source, mapping in [("hf_nickparvar", {"glioma": "2-glioma", "meningioma": "3-meningioma", "notumor": "1-notumor", "pituitary": "4-pituitary"}),
                            ("sartaj", {"glioma": "glioma_tumor", "meningioma": "meningioma_tumor", "notumor": "no_tumor", "pituitary": "pituitary_tumor"})]:
        for label, folder in mapping.items():
            if source == "hf_nickparvar":
                listing = get(f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{hf_revision}/Testing/{folder}?limit=1000").json()
                paths = sorted(x["path"] for x in listing if x["type"] == "file" and x["path"].lower().endswith((".jpg", ".png")))
            else:
                paths = sorted(x["path"] for x in tree["tree"] if x["path"].startswith(f"Testing/{folder}/") and x["path"].lower().endswith((".jpg", ".png")))
            for path in sorted(rng.sample(paths, min(args.per_class_per_source, len(paths)))):
                url = (f"https://huggingface.co/datasets/{HF_REPO}/resolve/{hf_revision}/{quote(path)}" if source == "hf_nickparvar" else
                       f"https://raw.githubusercontent.com/{GH_REPO}/{gh_revision}/{quote(path)}")
                records.append({"source": source, "source_revision": hf_revision if source == "hf_nickparvar" else gh_revision,
                                "label": label, "relative_path": f"images/{label}/{source}_{Path(path).name}",
                                "original_path": path, "download_url": url,
                                "source_page": f"https://huggingface.co/datasets/{HF_REPO}" if source == "hf_nickparvar" else f"https://github.com/{GH_REPO}",
                                "label_origin": "Publisher class folder; not independently reviewed by a radiologist"})
                old = previous_images.get(records[-1]["relative_path"], {})
                if old.get("download_url") == url:
                    records[-1]["expected_sha256"] = old.get("sha256")
    DEST.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(download, records))
    manifest = {"sample_seed": 20261101, "selection": "Uniform random sample from each source/class Testing directory before inference",
                "warning": "These sources share ancestry with the local dataset. Downloads are NOT an independent clinical validation set.",
                "sources": [{"name": "Simezu mirror of Masoud Nickparvar", "url": f"https://huggingface.co/datasets/{HF_REPO}",
                             "license_note": "Mirror attributes Kaggle/Figshare/SARTAJ/Br35H; consult upstream terms before redistribution."},
                            {"name": "Sartaj Bhuvaji", "url": f"https://github.com/{GH_REPO}",
                             "license_note": "No license asserted here; retained locally for requested research testing; review upstream terms before redistribution."}],
                "images": downloaded}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded and decoded {len(downloaded)} labeled MRI images to {DEST}", flush=True)


if __name__ == "__main__":
    main()
