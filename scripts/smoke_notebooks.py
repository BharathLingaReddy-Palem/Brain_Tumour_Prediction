"""Execute setup/data/model cells without retraining or overwriting saved artifacts."""
import contextlib
from io import StringIO
import json
from pathlib import Path
import sys

from IPython.core.interactiveshell import InteractiveShell
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
shell = InteractiveShell.instance()


def run_cells(path, indices):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"display": lambda *args, **kwargs: None}
    for index in indices:
        source = "".join(notebook["cells"][index]["source"])
        code = shell.input_transformer_manager.transform_cell(source)
        exec(compile(code, f"{path.name}:cell{index}", "exec"), namespace)
    return namespace


def main():
    results = []
    log = StringIO()
    with contextlib.redirect_stdout(log):
        p = ROOT/"brain_tumor_cnn_step1_2 (1).ipynb"
        ns = run_cells(p, [1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 15])
        model = ns["build_basic_cnn"](4).eval()
        with torch.no_grad():
            shape = list(model(ns["images"][:2]).shape)
        assert shape == [2, 4]
        results.append({"notebook": p.name, "status": "passed", "output_shape": shape,
                        "scope": "setup, class checks, data loaders, one batch, BasicCNN forward; no training/export"})
        p = ROOT/"hyperparameter_tuning_brain_tumor (4).ipynb"
        ns = run_cells(p, [2, 3, 4, 5, 6, 7])
        train, val, test, classes = ns["make_data"](batch_size=2)
        image, _ = next(iter(val))
        with torch.no_grad():
            shape = list(ns["build_model"]("BasicCNN", len(classes)).eval()(image).shape)
        assert shape == [2, 4]
        results.append({"notebook": p.name, "status": "passed", "output_shape": shape,
                        "scope": "setup, Optuna import, transforms, data loaders, BasicCNN forward; no study/retraining/export"})
        p = ROOT/"brain_tumor_inference.ipynb"
        ns = run_cells(p, [1, 2, 3, 4, 5])
        image = sorted((ROOT/"Dataset/Testing/glioma").glob("*.jpg"))[0]
        label, confidence, probs = ns["predict_image"](image)
        assert label in probs and abs(sum(probs.values())-1) < 1e-5
        results.append({"notebook": p.name, "status": "passed", "sample_prediction": label,
                        "scope": "setup, checkpoint load, one real image prediction; full held-out evaluation in evaluate_model.py"})
    (ROOT/"artifacts/audit/notebook_smoke.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ROOT/"artifacts/audit/notebook_smoke.log").write_text(log.getvalue(), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
