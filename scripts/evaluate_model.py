"""Reproducible local/web evaluation using exactly the application's inference module."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch

from image_io import decode_image
from inference import load_inference_artifacts


def metrics(rows, labels):
    if not rows:
        return {"count": 0, "accuracy": None, "note": "No eligible images; accuracy cannot be estimated."}
    true = [r["true_label"] for r in rows]
    pred = [r["prediction"] for r in rows]
    n = len(rows)
    acc = accuracy_score(true, pred)
    z = 1.96
    center = (acc + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
    report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    return {"count": n, "correct": sum(a == b for a, b in zip(true, pred)), "accuracy": acc,
            "wilson_95_interval": [center-half, center+half], "macro_f1": report["macro avg"]["f1-score"],
            "classification_report": report, "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist()}


def evaluate(records, model, device, mapping, transform, batch_size):
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset+batch_size]
            images = torch.stack([transform(decode_image(r["path"])) for r in batch]).to(device)
            probabilities = model(images).softmax(dim=1).cpu().numpy()
            for r, probs in zip(batch, probabilities):
                idx = int(probs.argmax())
                row = {k: v for k, v in r.items() if k != "path"}
                row.update(image_path=str(r["path"].relative_to(ROOT)), prediction=mapping[idx], confidence=float(probs[idx]),
                           correct=mapping[idx] == r["true_label"])
                row.update({f"prob_{mapping[i]}": float(v) for i, v in enumerate(probs)})
                rows.append(row)
            if offset % (batch_size*10) == 0:
                print(f"Evaluated {min(offset+batch_size,len(records))}/{len(records)}", flush=True)
    return rows, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    index = json.loads((ROOT/"artifacts/audit/dataset_index.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT/"webtesting_folder/manifest.json").read_text(encoding="utf-8"))
    model_path = ROOT/"artifacts/best_brain_tumor_model.pth"
    model, device, mapping, transform, name = load_inference_artifacts(model_path, ROOT/"artifacts/class_mapping.json")
    labels = list(mapping.values())
    train_hashes = {r["pixel_sha256"] for r in index if r["split"] == "Training"}
    local_hashes = {r["pixel_sha256"] for r in index}
    test_records = [{"path": ROOT/r["path"], "true_label": r["label"], "source": "local_testing",
                     "overlaps_training": r["pixel_sha256"] in train_hashes, "pixel_sha256": r["pixel_sha256"]}
                    for r in index if r["split"] == "Testing"]
    web_records = [{"path": ROOT/"webtesting_folder"/r["relative_path"], "true_label": r["label"], "source": r["source"],
                    "overlaps_training": r["pixel_sha256"] in train_hashes, "overlaps_any_local": r["pixel_sha256"] in local_hashes,
                    "pixel_sha256": r["pixel_sha256"], "download_url": r["download_url"]} for r in manifest["images"]]
    local_rows, local_seconds = evaluate(test_records, model, device, mapping, transform, args.batch_size)
    web_rows, web_seconds = evaluate(web_records, model, device, mapping, transform, args.batch_size)
    # Deduplicate each reported clean slice by decoded pixels, including relabeled duplicates.
    def unique(rows):
        seen, result = set(), []
        for row in rows:
            if row["pixel_sha256"] not in seen:
                result.append(row)
                seen.add(row["pixel_sha256"])
        return result
    clean_local = unique([r for r in local_rows if not r["overlaps_training"]])
    clean_web = unique([r for r in web_rows if not r["overlaps_any_local"]])
    report = {"checkpoint": str(model_path.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "model_name": name, "device": str(device), "torch_version": str(torch.__version__), "python": sys.executable,
              "preprocessing": "RGB; resize 224x224 bilinear; ToTensor [0,1]; no ImageNet normalization",
              "labels": labels, "local_testing": metrics(local_rows, labels),
              "local_unique_without_training_pixel_overlap": metrics(clean_local, labels),
              "web_all": metrics(web_rows, labels),
              "web_by_source": {s: metrics([r for r in web_rows if r["source"] == s], labels) for s in sorted({r["source"] for r in web_rows})},
              "web_unique_without_any_local_pixel_overlap": metrics(clean_web, labels),
              "overlap": {"web_training_matches": sum(r["overlaps_training"] for r in web_rows),
                          "web_any_local_matches": sum(r["overlaps_any_local"] for r in web_rows)},
              "inference_seconds": {"local": local_seconds, "web": web_seconds},
              "limitations": ["Web sources share ancestry with training data; even pixel-unmatched images are not proven independent.",
                              "Exact decoded-pixel comparison does not detect all resizes, compression variants, or repeated patients.",
                              "Folder labels are publisher labels, not newly confirmed pathology.",
                              "Wilson intervals describe image-level counts and assume independence; repeated patients violate that assumption.",
                              "No medical diagnostic accuracy or segmentation accuracy is established by this test."]}
    out = ROOT/"webtesting_folder/results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(local_rows).to_csv(out/"local_predictions.csv", index=False)
    pd.DataFrame(web_rows).to_csv(out/"web_predictions.csv", index=False)
    (out/"evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for title, rows, filename in [("Local Testing (original split)", local_rows, "local_confusion_matrix.png"),
                                  ("Web downloads (shared dataset ancestry)", web_rows, "web_confusion_matrix.png")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions([r["true_label"] for r in rows], [r["prediction"] for r in rows], labels=labels, ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out/filename, dpi=160)
        plt.close(fig)
    print(json.dumps({k: {x: v[x] for x in ("count", "accuracy", "macro_f1") if x in v}
                      for k, v in report.items() if isinstance(v, dict) and "accuracy" in v}, indent=2), flush=True)


if __name__ == "__main__":
    main()
