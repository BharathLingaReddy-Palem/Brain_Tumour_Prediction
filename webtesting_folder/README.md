# Downloaded MRI testing

This folder contains 200 downloaded, publisher-labeled MRIs: 25 per class from each of two sources. The random seed is `20261101`. Selection happened before inference, using only the source folders and seed.

- [Simezu mirror of the Nickparvar dataset](https://huggingface.co/datasets/Simezu/brain-tumour-MRI-scan): 100 images.
- [Sartaj Bhuvaji original classification dataset](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet): 100 images.

`manifest.json` records the original class, pinned source revision, download URL, SHA-256, and decoded-pixel hash for every image. Images are under `images/<class>/`. Source terms must be checked before redistribution; downloaded images are excluded from Git.

## Measured results

The existing checkpoint predicted 176/200 correctly (**88.0%**, macro F1 **0.8726**). The real browser upload processed and saved all 200, and its exported predictions exactly matched the offline evaluation.

This is an operational test, **not an independent medical validation**: 193 downloads exactly match decoded local dataset images, including 87 matching the training set. Seven unmatched downloaded files represent six unique decoded images; 4/6 were correct. Six images, shared dataset ancestry, and unknown patient overlap prevent a credible external accuracy estimate.

## Results files

- `results/evaluation.json`: full metrics, per-class reports, confusion matrices, overlap counts, limitations, and checkpoint hash.
- `results/web_predictions.csv`: predictions with true source labels, probabilities, and overlap flags.
- `results/local_predictions.csv`: all 1,600 local test predictions.
- `results/browser_batch_predictions.csv`: CSV actually downloaded from the running browser.
- `results/browser_verification.json`: browser/offline consistency and 88% accuracy check.
- `results/*confusion_matrix.png`: figures from the measured predictions.
- `results/browser_prediction.png`, `results/browser_batch.png`: browser evidence.

## Repeat using the requested environment

From the project root in PowerShell:

```powershell
$python = 'C:\Users\acer\anaconda3\envs\computer_vision\python.exe'
& $python scripts\audit_project.py
& $python scripts\download_web_test_images.py
& $python scripts\evaluate_model.py
& $python -m unittest discover -s tests -v
& $python scripts\smoke_notebooks.py
```

The downloader checks existing files and records hashes. The evaluator preserves the model and the original dataset. Tests use temporary SQLite databases and temporary uploads, including the synthetic ground-truth mask used to check segmentation input handling. That synthetic mask does not measure segmentation quality.
