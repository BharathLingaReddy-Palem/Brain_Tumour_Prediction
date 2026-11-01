"""Read-only inventory, image integrity, and exact-duplicate audit."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def main():
    inventory = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(x in rel.parts for x in (".git", "__pycache__")) or rel.parts[:2] == ("artifacts", "audit"):
            continue
        inventory.append({"path": rel.as_posix(), "bytes": p.stat().st_size})
    (OUT / "file_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    auxiliary_images = []
    for entry in inventory:
        path = ROOT / entry["path"]
        if path.suffix.lower() in EXTS and not entry["path"].startswith("Dataset/"):
            try:
                with Image.open(path) as image:
                    image.load()
                    auxiliary_images.append({"path": entry["path"], "size": list(image.size), "status": "readable"})
            except Exception as exc:
                auxiliary_images.append({"path": entry["path"], "error": str(exc)})
    (OUT / "auxiliary_images.json").write_text(json.dumps(auxiliary_images, indent=2), encoding="utf-8")
    records, bad = [], []
    for p in sorted((ROOT / "Dataset").rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as image:
                image.load()
                records.append({"path": p.relative_to(ROOT).as_posix(), "split": p.parent.parent.name,
                                "label": p.parent.name, "size": list(image.size),
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "pixel_sha256": pixel_hash(image)})
        except Exception as exc:
            bad.append({"path": str(p), "error": str(exc)})
    groups = defaultdict(list)
    for r in records:
        groups[r["pixel_sha256"]].append(r)
    cross_split = [v for v in groups.values() if len({r["split"] for r in v}) > 1]
    conflicts = [v for v in groups.values() if len({r["label"] for r in v}) > 1]
    summary = {"python": sys.executable, "file_count": len(inventory), "image_count": len(records),
               "counts": dict(Counter(f'{r["split"]}/{r["label"]}' for r in records)),
               "corrupt_images": bad, "duplicate_pixel_groups": sum(len(v) > 1 for v in groups.values()),
               "cross_split_duplicate_groups": len(cross_split),
               "test_images_overlapping_training": sum(r["split"] == "Testing" for v in cross_split for r in v),
               "conflicting_label_groups": len(conflicts)}
    (OUT / "dataset_index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT / "dataset_integrity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "duplicate_details.json").write_text(json.dumps({"cross_split": cross_split, "label_conflicts": conflicts}, indent=2), encoding="utf-8")
    notebooks = []
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    for p in ROOT.glob("*.ipynb"):
        notebook = json.loads(p.read_text(encoding="utf-8"))
        errors, stored = [], []
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            try:
                ast.parse(shell.input_transformer_manager.transform_cell("".join(cell["source"])))
            except SyntaxError as exc:
                errors.append({"cell": i, "error": str(exc)})
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    stored.append({"cell": i, "error": output.get("ename"), "message": output.get("evalue")})
        notebooks.append({"path": p.name, "cells": len(notebook["cells"]), "syntax_errors": errors, "stored_errors": stored})
    (OUT / "notebook_audit.json").write_text(json.dumps(notebooks, indent=2), encoding="utf-8")
    docs = []
    from pypdf import PdfReader
    for p in ROOT.rglob("*.pdf"):
        try:
            reader = PdfReader(p)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            (OUT / (p.parent.name + "_" + p.stem + ".txt")).write_text(content, encoding="utf-8")
            docs.append({"path": p.relative_to(ROOT).as_posix(), "pages": len(reader.pages), "text_characters": len(content)})
        except Exception as exc:
            docs.append({"path": str(p), "error": str(exc)})
    for p in ROOT.glob("*.pptx"):
        with zipfile.ZipFile(p) as archive:
            slides = [n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            text = "\n".join(" ".join(ET.fromstring(archive.read(n)).itertext()) for n in sorted(slides))
            (OUT / (p.stem + ".txt")).write_text(text, encoding="utf-8")
            docs.append({"path": p.name, "slides": len(slides)})
    (OUT / "document_audit.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(notebooks, indent=2), flush=True)


if __name__ == "__main__":
    main()
