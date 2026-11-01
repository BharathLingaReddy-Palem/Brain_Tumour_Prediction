# Project audit and testing — 1 November 2026

## Outcome

Reviewed the application and all segmentation modules, the three notebooks, checkpoint structure and weights, saved CSVs, project documentation, and the SQLite schema/integrity. Inventoried all existing project files, decoded all 7,200 local dataset images, extracted text from all three PDFs and the 12-slide presentation, and examined the existing report sources. Model inference and application behavior were tested in `C:\Users\acer\anaconda3\envs\computer_vision\python.exe`.

The original user database and MRI dataset were preserved. All test accounts, saved predictions, and uploaded copies use separate test databases/directories. Model weights were not retrained or replaced. The previously modified database and existing untracked `cv_report/` and `pdf/` folders were retained.

## Classification measurements

| Evaluation | Correct / images | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Original local Testing split | 1,504 / 1,600 | 94.00% | 0.9384 |
| Unique local test images excluding exact training pixel overlap | 1,388 / 1,468 | 94.55% | 0.9425 |
| All web downloads | 176 / 200 | 88.00% | 0.8726 |
| Nickparvar mirror web sample | 98 / 100 | 98.00% | 0.9800 |
| Sartaj web sample | 78 / 100 | 78.00% | 0.7378 |
| Unique web images with no exact local pixel match | 4 / 6 | 66.67% | 0.5000 |

Removing overlap changes the evaluated population; the 94.55% value is not an improvement to the model. The final row is too small and too closely related to the existing data sources to establish generalization. Exact pixel matching misses recompression, resizing, and shared-patient variants.

The browser accepted all 200 downloaded images, saved 200 results, and exported predictions identical to the evaluation script: 176/200 correct. The browser used a separate local test database. The manifest pins source revisions and records the publisher labels, URLs, and image hashes. Results are under `webtesting_folder/results/`.

The deployment and tuned checkpoints contain identical model weights. The deployed Basic CNN has 26,081,092 parameters and expects RGB images resized to 224 × 224, followed by conversion to [0,1] tensors. Existing weights were trained without ImageNet normalization; changing inference normalization alone would invalidate that training contract.

Checkpoint SHA-256: `6d8dc97ed011dc223eb38abeec8c4e40723d445dad228ef3e4f071993358bbe9`.

### Class-level weakness

| Local class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Glioma | 98.47% | 80.50% | 88.58% |
| Meningioma | 91.17% | 95.50% | 93.28% |
| No tumor | 91.12% | 100.00% | 95.35% |
| Pituitary | 96.39% | 100.00% | 98.16% |

Glioma accounts for 78 of the 96 local errors: 37 were predicted as meningioma, 33 as no tumor, and eight as pituitary. Thirty-one local errors and 11 web errors had scores of at least 85%. Therefore a high softmax score is not evidence of correctness. Neither these dataset results nor the saliency overlays establish clinical reliability.

## Errors fixed

1. **Ground-truth uploads crashed:** segmentation passed raw bytes to `Image.open` instead of a byte stream. All upload paths now share a fully decoding image reader with actionable validation errors.
2. **Mask export could not round-trip:** the “binary” PNG used red foreground. Grayscale thresholding erased it. Exports now contain actual grayscale 0/255 masks; ingestion accepts 0/1 masks too.
3. **Validation/test augmentation leakage:** PyTorch subsets shared one mutable dataset, so setting `train.augment=True` changed all splits. The training subset now uses its own dataset copy; evaluation remains deterministic.
4. **Intensity corruption in augmentation:** brightness augmentation clipped normalized [-1,1] MRI values to [0,1], destroying negative intensities. It now preserves the normalization range.
5. **Missing masks silently became healthy examples:** paired training/evaluation datasets now fail on missing ground truth. Zero masks require explicit inference-only opt-in.
6. **Admin filters crashed:** partial date ranges indexed an absent end date, regex metacharacters were interpreted as patterns, and reset changed already-instantiated widget state. Date validation, literal searches, and a reset callback fix these paths.
7. **Failed batches reported success:** batches containing unreadable images no longer report high-confidence success when all files failed.
8. **Saliency modified shared model gradients:** SmoothGrad now requests gradients only for the input and holds the target class constant across noisy samples. This avoids parameter-gradient accumulation and mixing explanations for different predicted classes.
9. **Repeated database-engine creation:** the engine is cached, SQLite connections close between operations, and the default database path is anchored to the project directory instead of the shell working directory.
10. **Unsafe file deletion and orphaned scans:** deleting a record can only unlink files under the configured upload root. Admin account deletion now cleans up that user's upload directory too.
11. **Predictable bootstrap credentials:** fresh databases only create the bootstrap admin when `ADMIN_PASSWORD` is provided. Existing credentials are preserved; changing `ADMIN_PASSWORD` does not rotate existing accounts.
12. **Graph-cut bookkeeping:** the normalized-cut fallback now uses superpixel IDs rather than final cluster IDs when selecting superpixels, allows a valid zero-valued cluster inside the brain mask, and handles blank regions. The reported method identifies a fallback when one occurs.
13. **Invalid metric inputs:** shape mismatches, empty masks, and nonfinite masks fail clearly rather than giving misleading segmentation scores.
14. **Version and platform friction:** migrated current Streamlit calls to `width`, enforced a 32 MB upload limit, replaced deprecated colormap access, added the missing Optuna/requests requirements, constrained OpenCV for NumPy 1.26, used safe checkpoint loading, and set notebook DataLoader workers to zero on Windows to avoid notebook multiprocessing problems.
15. **Misleading wording:** the no-tumor guide no longer calls predictions healthy scans; the app explains that confidence is uncalibrated, graph-cut regions are experimental candidates, sizes are resized-image pixels, and left/right controls refer to the image rather than patient anatomy.
16. **Training split and selection leakage:** future classification notebook runs now reserve test pixels first, remove exact duplicates, and stratify the remaining training/validation data through `data_splits.py`. Baseline model selection now uses validation accuracy; test accuracy and Grad-CAM focus are retained only as diagnostics. The tuning notebook no longer suppresses all warnings. These changes do not retroactively clean the saved checkpoint's training history.
17. **Test discovery:** removed the Git ignore rule that excluded every `test_*.py` file. Regression tests can now be versioned.

## What was validated

- **24 regression tests passed** (`python -m unittest discover -s tests -v`), plus the three notebook smoke workflows and the independent real-browser checks.
- Application startup, login rendering, correct/incorrect password checks, real single upload and browser heatmap.
- All 200 real web MRI files through both the application batch handler and the browser upload/CSV-download flow.
- Saved prediction history in disposable databases, ownership checks, protected external files, and admin deletion of disposable uploads.
- Corrupt upload handling, failed batch status, admin regex/date/reset cases, and segmentation with an optional synthetic ground-truth mask.
- Binary mask export/re-import, metric validation, brightness range, split-specific augmentation, and all three graph-cut modes.
- All three notebooks: syntax and stored-error checks plus executable setup/data/model/inference smoke tests. Full training and the 40-trial Optuna study were not rerun on this CPU-only environment.
- Local SQLite `PRAGMA integrity_check`: `ok`. All 255 auxiliary image files in the audit inventory also decoded successfully. No account details or password hashes were exported in the audit.

The machine reports CPU-only PyTorch. The tests use standard-library `unittest`; no package installation was needed. Logs and machine-readable audit files are in `artifacts/audit/`.

## File and artifact findings

- `app.py`: authentication, persistence, uploads, predictions, saliency, admin analytics, and graph-cut UI. Inference architecture/loading moved to `inference.py`; upload decoding moved to `image_io.py`.
- `segmentation/`: runnable intensity/graph-based segmentation, evaluation metrics, and supervised-dataset helpers. No trained segmentation checkpoint is included. Without independently annotated masks, segmentation accuracy is unknown. Bright normal tissue can also be highlighted.
- `Dataset/`: 5,600 training and 1,600 testing files, balanced across four classes; no unreadable images. Found 277 duplicate decoded-pixel groups, including 111 groups crossing splits and 114 test files that exactly match training. No conflicting labels among exact duplicates. Patient identities are unavailable, so patient separation is unverified.
- `artifacts/`: two equivalent Basic CNN checkpoints, class mapping, historical model-comparison and Optuna CSVs, and a prior batch prediction CSV. New measurements are separate; historical CSVs were not overwritten.
- Notebooks: setup and forward passes work in the requested environment. Future runs use exact-pixel deduplication and stratified classification splits; the saved checkpoint still reflects its original training experiment. Unknown patient overlap, same-source test reuse, and missing transfer-model-specific ImageNet preprocessing limit experimental validity. Correct these in a new training experiment with versioned artifacts; do not change preprocessing for the already trained weights.
- `README.md`, report PDF/TeX, and presentation: historical 92.56% metrics are not the current checkpoint's 94.00% measurement. The README now separates the two. Existing academic documents were reviewed as historical artifacts; their layout and compiled outputs were not regenerated.
- `pdf/cvproject.pdf` contains assignment/reference material, including an unrelated credit-risk report, rather than results from this classifier. `Generative_AI_Assignment.tex` is a separate assignment. Neither is an inference dependency.
- `uploaded_scans/`, local database, screenshots, and reports contain user/project state and were not used as labeled validation data. Downloaded source images stay out of Git; the manifest and reproducible code remain available.

## Remaining limits and improvements, in priority order

1. **Rebuild evaluation splits before claiming accuracy.** Obtain patient IDs, separate patients and sites, remove exact/near duplicates, and stratify by diagnosis. Use an untouched external cohort with verified labels. LGG helper splits are still image-level; use patient groups for actual segmentation training. Keep this audit's original measurements unchanged for traceability.
2. **Improve glioma recall.** Review the saved false negatives, particularly the 33 local gliomas predicted as no tumor. Train new versions with suitable MRI sequences, consistent labeling, class-aware objectives, and a patient-separated validation set. Select operating thresholds on validation data, never on this web sample.
3. **Calibrate and allow abstention.** Add reliability plots, multiclass Brier/log loss, validation-fitted temperature scaling, and an “uncertain / needs review” outcome. Separately test non-MRI photos, CT scans, screenshots, blank inputs, and unsupported sequences; image decoding alone is not MRI detection. [Calibration guidance](https://scikit-learn.org/stable/modules/calibration).
4. **Correct the transfer-learning experiment.** Use each pretrained backbone's documented normalization while retraining, include preprocessing metadata in each checkpoint, and use the same transform at inference. Fine-tune selected backbone layers and keep frozen BatchNorm behavior consistent. The current results do not establish that transfer learning is intrinsically inferior. [Torchvision ResNet preprocessing](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50).
5. **Make segmentation measurable.** Collect actual masks, preserve image/mask registration, split by patient, compare graph cut against a supervised U-Net baseline, and report Dice/IoU plus false-positive results on no-tumor scans. Export original-resolution masks with explicit spacing before adding millimeter/volume measurements.
6. **Add useful review features.** Prioritize DICOM/NIfTI support, sequence/orientation metadata, an image-quality gate, per-case reviewer corrections, case IDs, model/preprocessing version in CSV exports, and a downloadable report containing all probabilities and limitations. Add model performance/drift views backed by verified labels rather than displaying confidence as accuracy.
7. **Improve usability and performance.** Persist the single prediction across reruns, add batch progress/cancel controls and a failed-file summary, run actual batched tensors, and limit total batch memory. The 26-million-parameter dense head dominates model size; test global average pooling/distillation in a new model version.
8. **Harden deployment.** Rotate any previously provisioned default admin credentials through account management; add login throttling, stronger password policy, HTTPS/secrets configuration, retention controls, and database backups before multi-user deployment. Deployment/database credentials should remain outside the repository.

The shared `computer_vision` environment has pre-existing package metadata conflicts: OpenCV 4.13 requires NumPy >=2 while the environment has NumPy 1.26; unrelated facenet/TensorFlow/LangChain dependencies also disagree. The project's tested paths run, and `requirements.txt` now avoids the OpenCV/NumPy conflict for new installs. Other installed projects were not downgraded or altered. See `artifacts/audit/environment_conflicts.txt`; a clean project-specific lockfile/environment would improve reproducibility.

## Reproduce

```powershell
$python = 'C:\Users\acer\anaconda3\envs\computer_vision\python.exe'
& $python scripts\audit_project.py
& $python scripts\download_web_test_images.py
& $python scripts\evaluate_model.py
& $python scripts\smoke_notebooks.py
& $python -m unittest discover -s tests -v
& $python -m streamlit run app.py
```

`DATABASE_URL` and `UPLOADS_DIR` can point at disposable test locations. A first-time administrator requires `ADMIN_EMAIL`, `ADMIN_PASSWORD`, and optionally `ADMIN_NAME`. The initial audit preserved the public deployment. A subsequent publication update deployed the repaired application to the existing Streamlit Community Cloud URL; the README contains live verification screenshots and deployment details.
