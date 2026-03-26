# Brain Tumor Classification

End-to-end brain MRI tumor classification project with:

- Streamlit application with authentication, prediction, batch upload, heatmaps, and admin dashboard.
- Training and tuning notebooks (Optuna-based hyperparameter search).
- Inference notebook for standalone model prediction workflows.
- Dataset organized as ImageFolder-compatible train/test directories.

This project is intended for educational and research use only.

## Project Structure

```text
Brain_Tumor_Classification/
|- app.py
|- requirements.txt
|- README.md
|- brain_tumor_app.db
|- Dataset/
|  |- Training/
|  |  |- glioma/
|  |  |- meningioma/
|  |  |- pituitary/
|  |  |- notumor/
|  |- Testing/
|     |- glioma/
|     |- meningioma/
|     |- pituitary/
|     |- notumor/
|- artifacts/
|  |- best_brain_tumor_model.pth
|  |- best_tuned_model_state.pth
|  |- class_mapping.json
|  |- optuna_trials_summary (1).csv
|  |- model_comparison_metrics.csv
|  |- batch_predictions.csv
|- brain_tumor_inference.ipynb
|- hyperparameter_tuning_brain_tumor (4).ipynb
|- brain_tumor_cnn_step1_2 (1).ipynb
|- assets/
|  |- examples/
|- uploaded_scans/
```

## What The App Includes

The Streamlit app in `app.py` currently supports:

- User signup/login with PBKDF2 password hashing.
- Role-based access (`user`, `admin`) with database-backed persistence.
- Single-image prediction with confidence scoring and class probability chart.
- Batch prediction for multiple images with CSV export.
- Saliency heatmap visualization (SmoothGrad-style overlay).
- Saved prediction history with deletion and CSV download.
- Admin analytics dashboard:
  - user counts,
  - class distribution,
  - confidence trends,
  - global prediction record filtering/deletion,
  - image viewer and image downloads.
- Account management:
  - user self-delete with password confirmation,
  - admin credential updates,
  - admin-managed user updates/deletions.

## Model Artifacts and Their Purpose

There are two model checkpoints in `artifacts/`:

1. `best_tuned_model_state.pth`
   - Training/tuning artifact.
   - Useful for experimentation and reproducibility.
   - May include tuning metadata (best params, class names, etc.).

2. `best_brain_tumor_model.pth`
   - Deployment artifact used by the Streamlit app.
   - Contains app-expected fields such as `model_name`, `state_dict`, `idx_to_class`, and `img_size`.

Important:

- The app loads `artifacts/best_brain_tumor_model.pth`.
- Re-exporting tuned weights into this file changes format/packaging, not model quality.
- Recent verification showed both checkpoints produce the same test metrics in this workspace.

## Model Comparison Results

Using saved results from:

- `artifacts/model_comparison_metrics.csv`
- `artifacts/optuna_trials_summary (1).csv`

Best-performing model overall for classification accuracy and macro-F1 is **Basic CNN**.

### Test Set Metrics (`model_comparison_metrics.csv`)

- Basic CNN: test accuracy `0.9256`, test macro-F1 `0.9244`
- EfficientNet-B0: test accuracy `0.8363`, test macro-F1 `0.8333`
- ResNet50: test accuracy `0.8313`, test macro-F1 `0.8260`

### Optuna Validation Summary (`optuna_trials_summary (1).csv`)

- BasicCNN: best val macro-F1 `0.9626`, mean val macro-F1 `0.9341`
- EfficientNet-B0: best val macro-F1 `0.8993`, mean val macro-F1 `0.7958`
- ResNet50: best val macro-F1 `0.8904`, mean val macro-F1 `0.7139`

## Environment Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py
```

## Database and Auth Configuration

By default, the app uses local SQLite:

- `sqlite:///brain_tumor_app.db`

You can configure with environment variables:

- `DATABASE_URL` (for Postgres/other SQLAlchemy-compatible DB)
- `ADMIN_EMAIL` (default: `admin@brain.local`)
- `ADMIN_PASSWORD` (default: `admin123`)
- `ADMIN_NAME` (default: `System Admin`)

If using Postgres, `postgres://` is auto-normalized to `postgresql://` in the app.

## Notebook Workflows

### `hyperparameter_tuning_brain_tumor (4).ipynb`

- Runs Optuna hyperparameter tuning.
- Retrains best config.
- Evaluates on test set.
- Exports artifacts for app compatibility.

Use this notebook when you want to improve or retune model settings.

### `brain_tumor_inference.ipynb`

- Loads deployed artifact.
- Performs single and batch inference.
- Exports batch prediction CSV.

Use this notebook for model sanity checks and offline prediction tasks.

## Re-Export Tuned Model for App

If you tune in notebook and want app to use latest tuned weights:

1. Open `hyperparameter_tuning_brain_tumor (4).ipynb`.
2. Run the export cell that writes:
   - `artifacts/best_brain_tumor_model.pth`
   - `artifacts/class_mapping.json`
3. Restart or rerun Streamlit app.

## Common Issues

### Missing variables in notebook export cell

If kernel was restarted and notebook variables are gone, the export flow can load from saved tuned artifacts (if present) and still generate app-compatible output.

### Streamlit model load error

Check these files exist:

- `artifacts/best_brain_tumor_model.pth`
- `artifacts/class_mapping.json`

### Notebook and app show different behavior

Ensure both are using the same deployed checkpoint (`best_brain_tumor_model.pth`) and same class mapping.

## Requirements

Current project dependencies (from `requirements.txt`):

- numpy
- pandas
- pillow
- torch
- torchvision
- matplotlib
- scikit-learn
- jupyter
- streamlit
- sqlalchemy

## Disclaimer

This project is not a medical diagnosis system.
All predictions should be reviewed by qualified medical professionals.

