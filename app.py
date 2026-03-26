from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import hmac
from io import BytesIO
from pathlib import Path
import json
import os
import re
import secrets
import zipfile

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------- Page setup ----------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="B",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f766e 0%, #134e4a 55%, #0b3b39 100%);
        color: #f8fafc;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.2);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 0.45rem 0 0;
        color: #dff6f2;
        font-size: 1rem;
    }
    .info-card {
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: #f8fafc;
        margin-bottom: 0.8rem;
    }
    .small-note {
        font-size: 0.88rem;
        color: #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Constants ----------
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_brain_tumor_model.pth"
CLASS_MAP_PATH = ARTIFACTS_DIR / "class_mapping.json"
DEFAULT_DATASET_TEST = ROOT / "Dataset" / "Testing"
EXAMPLE_ROOT = ROOT / "assets" / "examples"
UPLOADS_DIR = ROOT / "uploaded_scans"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_HEIGHT_7_ROWS = 300

CLASS_INFO = {
    "glioma": {
        "title": "Glioma",
        "description": [
            "Tumor that starts in glial cells inside brain tissue.",
            "Often irregular or spread out and can be malignant.",
            "Simple: Tumor growing within the brain itself.",
        ],
    },
    "meningioma": {
        "title": "Meningioma",
        "description": [
            "Tumor that starts in the meninges (brain covering).",
            "Usually outside brain tissue and often benign.",
            "Simple: Tumor on brain surface, not inside brain tissue.",
        ],
    },
    "pituitary": {
        "title": "Pituitary Tumor",
        "description": [
            "Tumor in the pituitary gland at the center of the brain.",
            "Can affect hormones and sometimes vision.",
            "Simple: Tumor in the hormone control center.",
        ],
    },
    "notumor": {
        "title": "No Tumor (Normal)",
        "description": [
            "No abnormal growth detected.",
            "Brain structure appears normal.",
            "Simple: Healthy brain scan.",
        ],
    },
}


# ---------- Database ----------
Base = declarative_base()
PBKDF2_ITERATIONS = 260000
PBKDF2_SALT_BYTES = 16
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.70

db_url = os.getenv("DATABASE_URL", "sqlite:///brain_tumor_app.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
engine = create_engine(db_url, future=True, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True)
    age = Column(Integer, nullable=True)
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    image_name = Column(String(255), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probability_json = Column(Text, nullable=True)
    image_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="predictions")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    payload = b"$".join(
        [
            b"pbkdf2_sha256",
            str(PBKDF2_ITERATIONS).encode("ascii"),
            base64.b64encode(salt),
            base64.b64encode(digest),
        ]
    )
    return payload.decode("ascii")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        algorithm, iter_str, salt_b64, digest_b64 = hashed_password.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # Lightweight schema migration for existing DBs.
    insp = inspect(engine)
    prediction_cols = {col["name"] for col in insp.get_columns("predictions")} if insp.has_table("predictions") else set()
    if "image_path" not in prediction_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN image_path TEXT"))

    admin_email = os.getenv("ADMIN_EMAIL", "admin@brain.local")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_name = os.getenv("ADMIN_NAME", "System Admin")

    with SessionLocal() as db:
        admin = db.query(User).filter(User.email == admin_email).first()
        if admin is None:
            admin = User(
                email=admin_email,
                password_hash=hash_password(admin_password),
                full_name=admin_name,
                role="admin",
            )
            db.add(admin)
            db.commit()


@st.cache_resource(show_spinner=False)
def setup_db_once() -> bool:
    init_db()
    return True


def get_user_by_email(email: str):
    with SessionLocal() as db:
        return db.query(User).filter(User.email == email.lower().strip()).first()


def create_user(email: str, password: str, full_name: str, phone: str, age: int | None):
    email_clean = email.lower().strip()

    with SessionLocal() as db:
        exists = db.query(User).filter(User.email == email_clean).first()
        if exists:
            return False, "Email already registered."

        user = User(
            email=email_clean,
            password_hash=hash_password(password),
            full_name=full_name.strip(),
            phone=phone.strip() if phone else None,
            age=age,
            role="user",
        )
        db.add(user)
        db.commit()
        return True, "Account created successfully."


def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def save_prediction(
    user_id: int,
    image_name: str,
    predicted_class: str,
    confidence: float,
    prob_df: pd.DataFrame,
    image_path: str | None = None,
):
    prob_json = prob_df.to_json(orient="records")
    with SessionLocal() as db:
        row = Prediction(
            user_id=user_id,
            image_name=image_name,
            predicted_class=predicted_class,
            confidence=float(confidence),
            probability_json=prob_json,
            image_path=image_path,
        )
        db.add(row)
        db.commit()


def get_user_predictions(user_id: int):
    with SessionLocal() as db:
        rows = (
            db.query(Prediction)
            .filter(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
    return rows


def get_all_users():
    with SessionLocal() as db:
        rows = db.query(User).order_by(User.created_at.desc()).all()
    return rows


def get_all_predictions():
    with SessionLocal() as db:
        rows = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return rows


@st.cache_data(show_spinner=False)
def build_image_name_index():
    index: dict[str, list[str]] = {}

    search_roots = [DEFAULT_DATASET_TEST, UPLOADS_DIR]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                key = p.name.lower()
                index.setdefault(key, []).append(str(p))

    return index


def resolve_prediction_image_path(prediction: Prediction) -> str | None:
    # First preference: explicit stored path.
    if prediction.image_path:
        p = Path(prediction.image_path)
        if p.exists():
            return str(p)

    # Fallback for old rows created before image_path existed.
    idx = build_image_name_index()
    matches = idx.get(str(prediction.image_name).lower(), [])
    if matches:
        return matches[0]

    return None


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def store_uploaded_image(user_id: int, uploaded_file) -> Path:
    user_dir = UPLOADS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = sanitize_filename(uploaded_file.name)
    out_path = user_dir / f"{stamp}_{safe_name}"
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


def update_user_credentials(user_id: int, new_email: str | None = None, new_password: str | None = None):
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return False, "User not found."

        if new_email:
            email_clean = new_email.lower().strip()
            existing = db.query(User).filter(User.email == email_clean, User.id != user_id).first()
            if existing is not None:
                return False, "Email already in use by another account."
            user.email = email_clean

        if new_password:
            if len(new_password) < 6:
                return False, "Password must be at least 6 characters."
            user.password_hash = hash_password(new_password)

        db.commit()
        return True, "Account updated successfully."


def delete_prediction_record(prediction_id: int, requester_user_id: int, is_admin: bool):
    with SessionLocal() as db:
        row = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if row is None:
            return False, "Prediction record not found."

        if not is_admin and row.user_id != requester_user_id:
            return False, "Not authorized to delete this record."

        img_path = row.image_path
        db.delete(row)
        db.commit()

    if img_path:
        p = Path(img_path)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    return True, "Prediction record deleted."


def delete_all_user_predictions(user_id: int):
    with SessionLocal() as db:
        rows = db.query(Prediction).filter(Prediction.user_id == user_id).all()
        img_paths = [r.image_path for r in rows if r.image_path]
        for r in rows:
            db.delete(r)
        db.commit()

    for p in img_paths:
        fp = Path(p)
        if fp.exists():
            try:
                fp.unlink()
            except OSError:
                pass
    return len(rows)


def delete_user_account_by_admin(target_user_id: int, acting_admin_id: int):
    if target_user_id == acting_admin_id:
        return False, "Admin cannot delete their own account from this panel."

    with SessionLocal() as db:
        user = db.query(User).filter(User.id == target_user_id).first()
        if user is None:
            return False, "User not found."

        if user.role == "admin":
            admin_count = db.query(User).filter(User.role == "admin").count()
            if admin_count <= 1:
                return False, "Cannot delete the last admin account."

        db.delete(user)
        db.commit()

    return True, "User account deleted."


def delete_user_account_self(user_id: int, password: str):
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return False, "User not found."
        if user.role != "user":
            return False, "This action is available for user accounts only."
        if not verify_password(password, user.password_hash):
            return False, "Password is incorrect."

        db.delete(user)
        db.commit()

    user_dir = UPLOADS_DIR / f"user_{user_id}"
    if user_dir.exists():
        for p in user_dir.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        try:
            for sub in sorted(user_dir.rglob("*"), reverse=True):
                if sub.is_dir():
                    sub.rmdir()
            user_dir.rmdir()
        except OSError:
            pass

    return True, "Your account was deleted successfully."


# ---------- Model definitions ----------
class BasicCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_basic_cnn(num_classes: int) -> nn.Module:
    return BasicCNN(num_classes=num_classes)


def build_resnet50_transfer(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0_transfer(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model_name = checkpoint["model_name"]
    idx_to_class = checkpoint.get("idx_to_class", {})
    idx_to_class = {int(k): str(v) for k, v in idx_to_class.items()}

    if not idx_to_class and CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        from_json = class_map.get("idx_to_class", {})
        idx_to_class = {int(k): str(v) for k, v in from_json.items()}

    if not idx_to_class:
        raise ValueError("Could not load class mapping from checkpoint or class_mapping.json")

    num_classes = len(idx_to_class)
    img_size = int(checkpoint.get("img_size", 224))

    if model_name == "Basic CNN":
        model = build_basic_cnn(num_classes)
    elif model_name == "ResNet50":
        model = build_resnet50_transfer(num_classes)
    elif model_name == "EfficientNet-B0":
        model = build_efficientnet_b0_transfer(num_classes)
    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    infer_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    return model, device, idx_to_class, infer_transform, model_name


def predict_pil_image(pil_img: Image.Image):
    model, device, idx_to_class, infer_transform, _ = load_artifacts()

    x = infer_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    prob_df = pd.DataFrame(
        {
            "Class": [idx_to_class[i] for i in range(len(probs))],
            "Probability": [float(probs[i].item()) for i in range(len(probs))],
        }
    ).sort_values("Probability", ascending=False)

    return pred_label, pred_conf, prob_df


def confidence_alert_level(pred_conf: float):
    if pred_conf >= CONFIDENCE_HIGH:
        return "high", "High confidence"
    if pred_conf >= CONFIDENCE_MEDIUM:
        return "medium", "Moderate confidence"
    return "low", "Low confidence"


def build_saliency_overlay(pil_img: Image.Image):
    model, device, _, infer_transform, _ = load_artifacts()

    x0 = infer_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    _, _, h, w = x0.shape
    smooth = np.zeros((h, w), dtype=np.float32)

    # SmoothGrad produces more stable and visually meaningful saliency than raw single-pass gradients.
    smooth_samples = 10
    noise_std = 0.10
    for _ in range(smooth_samples):
        noisy = (x0 + noise_std * torch.randn_like(x0)).clamp(0.0, 1.0).detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(noisy)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred_idx]
        score.backward()

        grad = noisy.grad.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
        smooth += grad.astype(np.float32)

    smooth /= float(smooth_samples)

    # Robust percentile normalization makes weak-but-relevant regions visible.
    p_low = float(np.percentile(smooth, 2.0))
    p_high = float(np.percentile(smooth, 99.5))
    if p_high > p_low:
        smooth = np.clip((smooth - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        smooth = np.zeros_like(smooth)

    saliency_map = Image.fromarray((smooth * 255.0).astype("uint8"), mode="L").resize(pil_img.size)
    heat = np.array(saliency_map, dtype=np.float32) / 255.0

    base = np.array(pil_img.convert("RGB"), dtype=np.float32)
    cmap_rgb = (cm.get_cmap("turbo")(heat)[..., :3] * 255.0).astype(np.float32)

    # Stronger adaptive alpha to clearly differentiate overlay from original.
    alpha = (0.15 + 0.75 * np.power(heat, 0.8))[..., None]
    overlay = (base * (1.0 - alpha) + cmap_rgb * alpha).clip(0, 255).astype(np.uint8)

    heat_color = Image.fromarray(cmap_rgb.astype(np.uint8))
    return Image.fromarray(overlay), heat_color


def get_example_images(class_name: str, max_images: int = 3):
    found = []
    source_paths = [EXAMPLE_ROOT / class_name, DEFAULT_DATASET_TEST / class_name]

    for src in source_paths:
        if not src.exists():
            continue
        candidates = [p for p in sorted(src.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        found.extend(candidates)
        if len(found) >= max_images:
            break

    unique = []
    seen = set()
    for p in found:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique[:max_images]


def build_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def ensure_state():
    if "auth_user_id" not in st.session_state:
        st.session_state.auth_user_id = None
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if "auth_email" not in st.session_state:
        st.session_state.auth_email = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "batch_image_bytes" not in st.session_state:
        st.session_state.batch_image_bytes = {}


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def show_image_compat(image_obj, **kwargs):
    try:
        st.image(image_obj, **kwargs)
    except TypeError:
        use_container_width = kwargs.pop("use_container_width", None)
        if use_container_width:
            kwargs["use_column_width"] = True
        st.image(image_obj, **kwargs)


def _normalize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        col_dtype = str(out[col].dtype)
        if col_dtype.startswith("datetime"):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            out[col] = out[col].fillna("")
        elif col_dtype.startswith("string") or col_dtype == "object":
            out[col] = out[col].fillna("").astype(str)
    return out


def dataframe_compat(df: pd.DataFrame, **kwargs):
    st.dataframe(_normalize_df_for_streamlit(df), **kwargs)


def data_editor_compat(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    safe_df = _normalize_df_for_streamlit(df)
    editor_fn = getattr(st, "data_editor", None)
    if editor_fn is None:
        editor_fn = getattr(st, "experimental_data_editor", None)

    if editor_fn is None:
        st.warning("Editable grid is unavailable in this Streamlit version; showing a read-only table.")
        dataframe_compat(safe_df, use_container_width=kwargs.get("use_container_width", True), height=kwargs.get("height"))
        return safe_df

    try:
        return editor_fn(safe_df, **kwargs)
    except TypeError:
        fallback = {
            "key": kwargs.get("key"),
            "use_container_width": kwargs.get("use_container_width", True),
            "height": kwargs.get("height"),
        }
        return editor_fn(safe_df, **{k: v for k, v in fallback.items() if v is not None})


def logout():
    st.session_state.auth_user_id = None
    st.session_state.auth_role = None
    st.session_state.auth_email = None
    rerun_app()


def render_guide_and_examples():
    st.markdown("---")
    st.subheader("Tumor Class Guide")
    for class_key in ["glioma", "meningioma", "pituitary", "notumor"]:
        block = CLASS_INFO[class_key]
        with st.expander(block["title"], expanded=(class_key in {"glioma", "notumor"})):
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            for line in block["description"]:
                st.write(f"- {line}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Example MRI Images")
    st.caption("If assets/examples/<class>/ is not available, the app uses Dataset/Testing/<class>/ when present.")
    for class_key in ["glioma", "meningioma", "pituitary", "notumor"]:
        title = CLASS_INFO[class_key]["title"]
        st.markdown(f"### {title}")
        samples = get_example_images(class_key, max_images=3)
        if not samples:
            st.markdown(
                '<p class="small-note">No example images found for this class yet.</p>',
                unsafe_allow_html=True,
            )
            continue
        cols = st.columns(len(samples))
        for i, p in enumerate(samples):
            with cols[i]:
                show_image_compat(str(p), caption=p.name, use_container_width=True)


def render_login_signup():
    st.markdown(
        """
        <div class="hero">
          <h1>Brain Tumor MRI Classifier</h1>
          <p>Sign in to save your prediction history and access personalized records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        st.subheader("Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", use_container_width=True)

        if submit_login:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.auth_user_id = user.id
                st.session_state.auth_role = user.role
                st.session_state.auth_email = user.email
                st.success("Login successful.")
                rerun_app()

    with signup_tab:
        st.subheader("Create Account")
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email", key="signup_email")
            phone = st.text_input("Phone (optional)")
            age = st.number_input("Age (optional)", min_value=0, max_value=120, value=0, step=1)
            password = st.text_input("Password", type="password", key="signup_pwd")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Create Account", use_container_width=True)

        if submit_signup:
            if not full_name.strip() or not email.strip() or not password:
                st.error("Full name, email, and password are required.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                age_val = int(age) if age > 0 else None
                ok, msg = create_user(email=email, password=password, full_name=full_name, phone=phone, age=age_val)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

def render_user_prediction_page(current_user: User):
    st.markdown(
        """
        <div class="hero">
          <h1>Brain Tumor MRI Classifier</h1>
          <p>Upload an MRI image to predict one of four classes: Glioma, Meningioma, Pituitary, or No Tumor.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Upload MRI")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            key="user_uploader",
        )
        show_heatmap = st.checkbox("Show visual explanation (heatmap)", value=True, key="single_heatmap")
        run_pred = st.button("Predict", type="primary", use_container_width=True)
        if uploaded_file is not None:
            preview = Image.open(uploaded_file).convert("RGB")
            show_image_compat(preview, caption="Uploaded MRI", use_container_width=True)

    with right:
        st.subheader("Prediction Output")
        if run_pred:
            if uploaded_file is None:
                st.warning("Please upload an image first.")
            else:
                try:
                    uploaded_bytes = uploaded_file.getvalue()
                    pil = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
                    pred_label, pred_conf, prob_df = predict_pil_image(pil)

                    conf_pct = pred_conf * 100.0
                    level, level_text = confidence_alert_level(pred_conf)
                    if level == "high":
                        st.success(f"Prediction: {pred_label} ({conf_pct:.2f}%) - {level_text}")
                    elif level == "medium":
                        st.warning(f"Prediction: {pred_label} ({conf_pct:.2f}%) - {level_text}")
                        st.caption("Review is recommended before relying on this result.")
                    else:
                        st.error(f"Prediction: {pred_label} ({conf_pct:.2f}%) - {level_text}")
                        st.caption("Low confidence alert: clinical verification is strongly recommended.")

                    st.bar_chart(prob_df.set_index("Class"))

                    if show_heatmap:
                        try:
                            overlay_img, heat_img = build_saliency_overlay(pil)
                            h1, h2, h3 = st.columns(3)
                            with h1:
                                show_image_compat(pil, caption="Original MRI", use_container_width=True)
                            with h2:
                                show_image_compat(overlay_img, caption="Overlay explanation", use_container_width=True)
                            with h3:
                                show_image_compat(heat_img, caption="Saliency heatmap", use_container_width=True)
                        except Exception as hm_err:
                            st.info(f"Heatmap unavailable for this model output: {hm_err}")

                    saved_path = store_uploaded_image(current_user.id, uploaded_file)

                    save_prediction(
                        user_id=current_user.id,
                        image_name=uploaded_file.name,
                        predicted_class=pred_label,
                        confidence=pred_conf,
                        prob_df=prob_df,
                        image_path=str(saved_path),
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Batch Upload")
    st.caption("Upload multiple MRIs, run predictions in one click, and review confidence alerts.")

    batch_files = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )
    batch_heatmap = st.checkbox("Enable visual explanation selector for batch results", value=False, key="batch_heatmap")
    run_batch = st.button("Run Batch Prediction", type="primary", use_container_width=True, key="run_batch_btn")

    if run_batch:
        if not batch_files:
            st.warning("Please upload one or more images for batch prediction.")
        else:
            batch_rows = []
            batch_img_map: dict[str, bytes] = {}
            low_conf_count = 0
            medium_conf_count = 0

            for idx, bf in enumerate(batch_files, start=1):
                row_key = f"{idx:03d}_{sanitize_filename(bf.name)}"
                try:
                    file_bytes = bf.getvalue()
                    pil = Image.open(BytesIO(file_bytes)).convert("RGB")
                    pred_label, pred_conf, prob_df = predict_pil_image(pil)

                    level, level_text = confidence_alert_level(pred_conf)
                    if level == "low":
                        low_conf_count += 1
                    elif level == "medium":
                        medium_conf_count += 1

                    saved_path = store_uploaded_image(current_user.id, bf)
                    save_prediction(
                        user_id=current_user.id,
                        image_name=bf.name,
                        predicted_class=pred_label,
                        confidence=pred_conf,
                        prob_df=prob_df,
                        image_path=str(saved_path),
                    )

                    batch_img_map[row_key] = file_bytes
                    batch_rows.append(
                        {
                            "batch_key": row_key,
                            "file_name": bf.name,
                            "predicted_class": pred_label,
                            "confidence": round(float(pred_conf), 4),
                            "confidence_level": level_text,
                            "status": "saved",
                        }
                    )
                except Exception as err:
                    batch_rows.append(
                        {
                            "batch_key": row_key,
                            "file_name": bf.name,
                            "predicted_class": "-",
                            "confidence": 0.0,
                            "confidence_level": "error",
                            "status": f"failed: {err}",
                        }
                    )

            st.session_state.batch_results = batch_rows
            st.session_state.batch_image_bytes = batch_img_map

            if low_conf_count > 0:
                st.error(f"Low confidence alerts: {low_conf_count} case(s).")
            elif medium_conf_count > 0:
                st.warning(f"Moderate confidence alerts: {medium_conf_count} case(s).")
            else:
                st.success("Batch completed with high-confidence results.")

    if st.session_state.batch_results:
        batch_df = pd.DataFrame(st.session_state.batch_results)
        dataframe_compat(batch_df, use_container_width=True, height=TABLE_HEIGHT_7_ROWS)
        st.download_button(
            label="Download Batch Results CSV",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if batch_heatmap:
            valid_keys = [
                r["batch_key"]
                for r in st.session_state.batch_results
                if r.get("batch_key") in st.session_state.batch_image_bytes
            ]
            if valid_keys:
                selected_key = st.selectbox("Select batch image for heatmap", options=valid_keys, key="batch_heatmap_key")
                img_bytes = st.session_state.batch_image_bytes.get(selected_key)
                if img_bytes:
                    try:
                        sel_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        overlay_img, heat_img = build_saliency_overlay(sel_img)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            show_image_compat(sel_img, caption=f"Original ({selected_key})", use_container_width=True)
                        with c2:
                            show_image_compat(overlay_img, caption="Overlay explanation", use_container_width=True)
                        with c3:
                            show_image_compat(heat_img, caption="Saliency heatmap", use_container_width=True)
                    except Exception as hm_err:
                        st.info(f"Batch heatmap unavailable: {hm_err}")

    st.markdown("---")
    st.subheader("Prediction History")
    db_rows = get_user_predictions(current_user.id)
    if db_rows:
        db_data = [
            {
                "id": r.id,
                "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": r.image_name,
                "predicted_class": r.predicted_class,
                "confidence": round(float(r.confidence), 4),
            }
            for r in db_rows
        ]
        db_df = pd.DataFrame(db_data)
        dataframe_compat(db_df, use_container_width=True, height=TABLE_HEIGHT_7_ROWS)
        st.download_button(
            label="Download History CSV",
            data=db_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### Delete Records")
        delete_df = db_df[["id", "timestamp", "file_name", "predicted_class", "confidence"]].copy()
        delete_df.insert(0, "delete", False)
        edited_user_df = data_editor_compat(
            delete_df,
            hide_index=True,
            height=TABLE_HEIGHT_7_ROWS,
            use_container_width=True,
            key="user_delete_editor",
            column_config={
                "delete": st.column_config.CheckboxColumn("Delete"),
                "id": st.column_config.NumberColumn("ID", disabled=True),
            },
            disabled=["id", "timestamp", "file_name", "predicted_class", "confidence"],
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Delete Selected", use_container_width=True, type="secondary"):
                ids = edited_user_df.loc[edited_user_df["delete"] == True, "id"].astype(int).tolist()
                if not ids:
                    st.info("Select at least one row using the Delete checkbox.")
                else:
                    deleted_count = 0
                    for rec_id in ids:
                        ok, _ = delete_prediction_record(
                            prediction_id=rec_id,
                            requester_user_id=current_user.id,
                            is_admin=False,
                        )
                        if ok:
                            deleted_count += 1
                    st.success(f"Deleted {deleted_count} record(s).")
                    rerun_app()
        with col_b:
            if st.button("Delete All My History", use_container_width=True):
                deleted = delete_all_user_predictions(current_user.id)
                st.success(f"Deleted {deleted} records.")
                rerun_app()
    else:
        st.info("No saved predictions yet.")


def render_admin_page(current_admin: User):
    st.title("Admin Dashboard")
    st.caption("Admin can view all users and all saved prediction records.")

    users = get_all_users()
    preds = get_all_predictions()

    users_by_id = {u.id: u for u in users}

    st.subheader("Analytics Dashboard")
    total_cases = len(preds)
    tumor_counts = {
        "glioma": 0,
        "meningioma": 0,
        "pituitary": 0,
        "notumor": 0,
    }
    for p in preds:
        key = str(p.predicted_class).lower()
        if key in tumor_counts:
            tumor_counts[key] += 1

    m1, m2, m3 = st.columns(3)
    m1.metric("👥 Total Users", len(users))
    m2.metric("🏥 Total Cases", total_cases)
    m3.metric("🔐 Admins", len([u for u in users if u.role == "admin"]))

    st.markdown("""<div style='background: linear-gradient(90deg, #0f766e 0%, #134e4a 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
    <h3 style='color: #f0f9ff; margin: 0 0 1rem 0;'>📊 Tumor Classification Breakdown</h3>
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
    """ + f"""    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center;'>
        <div style='color: #dff6f2; font-size: 2rem; font-weight: bold;'>{tumor_counts['glioma']}</div>
        <div style='color: #a7f3d0; font-size: 0.9rem;'>Glioma ({tumor_counts['glioma']*100//max(total_cases,1)}%)</div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center;'>
        <div style='color: #dff6f2; font-size: 2rem; font-weight: bold;'>{tumor_counts['meningioma']}</div>
        <div style='color: #a7f3d0; font-size: 0.9rem;'>Meningioma ({tumor_counts['meningioma']*100//max(total_cases,1)}%)</div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center;'>
        <div style='color: #dff6f2; font-size: 2rem; font-weight: bold;'>{tumor_counts['pituitary']}</div>
        <div style='color: #a7f3d0; font-size: 0.9rem;'>Pituitary ({tumor_counts['pituitary']*100//max(total_cases,1)}%)</div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center;'>
        <div style='color: #dff6f2; font-size: 2rem; font-weight: bold;'>{tumor_counts['notumor']}</div>
        <div style='color: #a7f3d0; font-size: 0.9rem;'>No Tumor ({tumor_counts['notumor']*100//max(total_cases,1)}%)</div>
    </div>
    </div>
    </div>""" + """</div>""", unsafe_allow_html=True)

    plot_left, plot_right = st.columns(2)
    with plot_left:
        st.markdown("<h4 style='color: #0f766e; margin-bottom: 1rem;'>📈 Cases by Tumor Type</h4>", unsafe_allow_html=True)
        class_count_df = pd.DataFrame([tumor_counts]).T.rename(columns={0: "count"})
        st.bar_chart(class_count_df, color=["#14b8a6"])

    with plot_right:
        st.markdown("<h4 style='color: #0f766e; margin-bottom: 1rem;'>⭐ Average Confidence by Type</h4>", unsafe_allow_html=True)
        conf_buckets = {"glioma": [], "meningioma": [], "pituitary": [], "notumor": []}
        for p in preds:
            key = str(p.predicted_class).lower()
            if key in conf_buckets:
                conf_buckets[key].append(float(p.confidence))

        conf_avg = {
            k: (sum(v) / len(v) if v else 0.0)
            for k, v in conf_buckets.items()
        }
        conf_df = pd.DataFrame([conf_avg]).T.rename(columns={0: "avg_confidence"})
        st.bar_chart(conf_df, color=["#0d9488"])

    st.markdown("---")
    st.subheader("User Activity Tracking")
    if preds:
        activity_rows = []
        by_user: dict[int, list[Prediction]] = {}
        for p in preds:
            by_user.setdefault(p.user_id, []).append(p)
        for user_id, records in by_user.items():
            user_obj = users_by_id.get(user_id)
            latest = max(r.created_at for r in records)
            activity_rows.append(
                {
                    "user_id": user_id,
                    "email": user_obj.email if user_obj else "<deleted user>",
                    "full_name": user_obj.full_name if user_obj else "-",
                    "times_used": len(records),
                    "last_used": latest.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        activity_df = pd.DataFrame(activity_rows).sort_values("times_used", ascending=False)
        dataframe_compat(activity_df, use_container_width=True, height=TABLE_HEIGHT_7_ROWS)
    else:
        st.info("No activity yet.")

    st.markdown("---")
    st.subheader("Users")
    if users:
        users_df = pd.DataFrame(
            [
                {
                    "id": u.id,
                    "full_name": u.full_name,
                    "email": u.email,
                    "phone": u.phone,
                    "age": u.age,
                    "role": u.role,
                    "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for u in users
            ]
        )
        dataframe_compat(users_df, use_container_width=True, height=TABLE_HEIGHT_7_ROWS)
        st.download_button(
            label="Download Users CSV",
            data=users_df.to_csv(index=False).encode("utf-8"),
            file_name="users.csv",
            mime="text/csv",
        )

        st.markdown("### Manage User Accounts")
        options = [f"{u.id} | {u.email} ({u.role})" for u in users]
        selected_user_label = st.selectbox("Select user", options=options, key="admin_manage_user")
        selected_user_id = int(selected_user_label.split(" | ")[0])
        selected_user = next((u for u in users if u.id == selected_user_id), None)

        if selected_user is not None:
            new_email = st.text_input("Update email", value=selected_user.email, key="admin_new_email")
            new_password = st.text_input("Set new password (leave blank to keep current)", type="password")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Update User Credentials", use_container_width=True):
                    ok, msg = update_user_credentials(
                        user_id=selected_user.id,
                        new_email=new_email.strip() if new_email.strip() else None,
                        new_password=new_password if new_password else None,
                    )
                    if ok:
                        st.success(msg)
                        rerun_app()
                    else:
                        st.error(msg)
            with c2:
                if st.button("Delete User Account", use_container_width=True):
                    ok, msg = delete_user_account_by_admin(
                        target_user_id=selected_user.id,
                        acting_admin_id=current_admin.id,
                    )
                    if ok:
                        st.success(msg)
                        rerun_app()
                    else:
                        st.error(msg)

        st.markdown("### Admin Account Settings")
        admin_new_email = st.text_input("Change my admin email", value=current_admin.email, key="admin_self_email")
        admin_new_password = st.text_input("Change my admin password (optional)", type="password", key="admin_self_pwd")
        if st.button("Update My Admin Account", use_container_width=True):
            ok, msg = update_user_credentials(
                user_id=current_admin.id,
                new_email=admin_new_email.strip() if admin_new_email.strip() else None,
                new_password=admin_new_password if admin_new_password else None,
            )
            if ok:
                st.success(msg)
                st.session_state.auth_email = admin_new_email.strip() if admin_new_email.strip() else current_admin.email
                rerun_app()
            else:
                st.error(msg)
    else:
        st.info("No users found.")

    st.markdown("---")
    st.subheader("All Prediction Records")
    if preds:
        pred_df = pd.DataFrame(
            [
                {
                    "id": p.id,
                    "patient_id": p.user_id,
                    "patient_name": users_by_id[p.user_id].full_name if p.user_id in users_by_id else "<deleted user>",
                    "user_email": users_by_id[p.user_id].email if p.user_id in users_by_id else "<deleted user>",
                    "image_name": p.image_name,
                    "predicted_class": p.predicted_class,
                    "confidence": round(float(p.confidence), 4),
                    "image_path": p.image_path,
                    "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for p in preds
            ]
        )
        
        # Filter section
        st.markdown("### 🔍 Filter Records")
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            selected_class = st.multiselect(
                "🧬 Tumor Type",
                options=["glioma", "meningioma", "pituitary", "notumor"],
                default=["glioma", "meningioma", "pituitary", "notumor"],
                key="filter_class"
            )
        
        with filter_col2:
            min_conf, max_conf = st.slider(
                "⭐ Confidence Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                key="filter_conf"
            )
        
        with filter_col3:
            email_search = st.text_input(
                "🔎 Search Email (partial match)",
                value="",
                placeholder="e.g., user@example.com",
                key="filter_email_search"
            )

        with filter_col4:
            patient_name_search = st.text_input(
                "👤 Search Patient Name",
                value="",
                placeholder="e.g., Bharath",
                key="filter_patient_name",
            )

        filter_col_pid, filter_col_date, filter_col_reset = st.columns([1.3, 2.7, 1])
        with filter_col_pid:
            patient_id_search = st.text_input(
                "🆔 Patient ID",
                value="",
                placeholder="e.g., 3",
                key="filter_patient_id",
            )
        with filter_col_date:
            date_range = st.date_input(
                "📅 Date Range",
                value=(pd.to_datetime(pred_df["created_at"]).min().date(), pd.to_datetime(pred_df["created_at"]).max().date()),
                key="filter_date"
            )
        with filter_col_reset:
            st.write("")
            st.write("")
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state.filter_class = ["glioma", "meningioma", "pituitary", "notumor"]
                st.session_state.filter_conf = (0.0, 1.0)
                st.session_state.filter_email_search = ""
                st.session_state.filter_patient_name = ""
                st.session_state.filter_patient_id = ""
                rerun_app()
        
        # Apply filters
        filtered_df = pred_df[
            (pred_df["predicted_class"].isin(selected_class)) &
            (pred_df["confidence"] >= min_conf) &
            (pred_df["confidence"] <= max_conf) &
            (pred_df["user_email"].str.contains(email_search, case=False, na=False) if email_search else True) &
            (pred_df["patient_name"].str.contains(patient_name_search, case=False, na=False) if patient_name_search else True) &
            (pred_df["patient_id"].astype(str).str.contains(patient_id_search, case=False, na=False) if patient_id_search else True) &
            (pd.to_datetime(pred_df["created_at"]).dt.date >= date_range[0]) &
            (pd.to_datetime(pred_df["created_at"]).dt.date <= date_range[1])
        ]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(pred_df)} records**")
        dataframe_compat(filtered_df, use_container_width=True, height=TABLE_HEIGHT_7_ROWS)
        st.download_button(
            label="📥 Download Filtered CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )

        st.markdown("### Delete Saved Record")
        admin_delete_df = filtered_df[["id", "patient_id", "patient_name", "user_email", "image_name", "predicted_class", "confidence", "created_at"]].copy()
        admin_delete_df.insert(0, "delete", False)
        edited_admin_df = data_editor_compat(
            admin_delete_df,
            hide_index=True,
            height=TABLE_HEIGHT_7_ROWS,
            use_container_width=True,
            key="admin_delete_editor",
            column_config={
                "delete": st.column_config.CheckboxColumn("Delete"),
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "patient_id": st.column_config.NumberColumn("Patient ID", disabled=True),
            },
            disabled=["id", "patient_id", "patient_name", "user_email", "image_name", "predicted_class", "confidence", "created_at"],
        )
        if st.button("Delete Selected Prediction(s)", use_container_width=True):
            ids = edited_admin_df.loc[edited_admin_df["delete"] == True, "id"].astype(int).tolist()
            if not ids:
                st.info("Select at least one row using the Delete checkbox.")
            else:
                deleted_count = 0
                for rec_id in ids:
                    ok, _ = delete_prediction_record(
                        prediction_id=rec_id,
                        requester_user_id=current_admin.id,
                        is_admin=True,
                    )
                    if ok:
                        deleted_count += 1
                st.success(f"Deleted {deleted_count} record(s).")
                rerun_app()

        st.markdown("### 🖼️ Image Storage Viewer")
        
        # Create all images ZIP
        all_images_zip = BytesIO()
        images_count = 0
        with zipfile.ZipFile(all_images_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for pred_record in preds:
                img_path = resolve_prediction_image_path(pred_record)
                if img_path and Path(img_path).exists():
                    try:
                        zip_file.write(img_path, arcname=Path(img_path).name)
                        images_count += 1
                    except Exception:
                        pass
        all_images_zip.seek(0)
        
        # Check if filtered_df has records
        if len(filtered_df) > 0:
            viewer_col1, viewer_col2, viewer_col3 = st.columns([2, 1, 1])
            with viewer_col1:
                viewer_pred_id = st.selectbox("Select prediction id to view image", options=filtered_df["id"].tolist(), key="admin_view_pred")
            
            with viewer_col2:
                st.write("")
                st.write("")
                st.caption("(Download below)")
            
            with viewer_col3:
                st.download_button(
                    label=f"📦 All ({images_count})",
                    data=all_images_zip.getvalue(),
                    file_name="all_brain_tumor_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="btn_download_all_active"
                )
            
            # Only process if viewer_pred_id is not None
            if viewer_pred_id is not None:
                selected_pred = next((p for p in preds if p.id == int(viewer_pred_id)), None)
                if selected_pred is not None:
                    st.write(f"📋 **Predicted class:** {selected_pred.predicted_class}")
                    st.write(f"⭐ **Confidence:** {selected_pred.confidence:.4f}")
                    resolved_path = resolve_prediction_image_path(selected_pred)
                    if resolved_path:
                        # Display image with reduced size
                        col_img, col_empty = st.columns([1, 2])
                        with col_img:
                            st.image(resolved_path, caption=Path(resolved_path).name, width=350)
                        
                        # Single image download button below image
                        col_single_dl, col_spacer = st.columns([1, 4])
                        with col_single_dl:
                            with open(resolved_path, "rb") as f:
                                st.download_button(
                                    label="💾 Download This Image",
                                    data=f.read(),
                                    file_name=Path(resolved_path).name,
                                    mime="image/jpeg",
                                    use_container_width=True,
                                    key=f"single_download_{selected_pred.id}"
                                )
                    else:
                        st.info("🚫 Image not found for this record. Older records created before image storage may not have a saved path.")
        else:
            st.info("📌 No records match your filters. Adjust the filter options to view images.")
    else:
        st.info("No predictions found.")


def get_current_user():
    user_id = st.session_state.auth_user_id
    if user_id is None:
        return None
    with SessionLocal() as db:
        return db.query(User).filter(User.id == int(user_id)).first()


def main():
    setup_db_once()
    ensure_state()

    current_user = get_current_user()

    with st.sidebar:
        st.header("Model Info")
        try:
            _model, _device, _idx_to_class, _transform, _model_name = load_artifacts()
            st.success(f"Loaded: {_model_name}")
            st.write(f"Device: {_device}")
            st.caption("Image tumor classification.")
            st.markdown("**Classes**")
            ordered_classes = [_idx_to_class[k] for k in sorted(_idx_to_class.keys())]
            for cls_name in ordered_classes:
                st.write(f"- {cls_name}")
        except Exception as e:
            st.error(f"Model load error: {e}")

        st.markdown("---")
        st.markdown("### Notes")
        st.write("This tool is for educational support and not a medical diagnosis.")
        

        if current_user is not None:
            st.markdown("---")
            st.write(f"Logged in as: {current_user.email}")
            st.write(f"Role: {current_user.role}")
            if st.button("Logout", use_container_width=True):
                logout()

            if current_user.role == "user":
                with st.expander("Danger Zone"):
                    st.caption("Delete your account permanently. This removes your prediction history and cannot be undone.")
                    confirm_delete = st.checkbox("I understand this action is permanent.", key="confirm_delete_account")
                    delete_pwd = st.text_input("Confirm password", type="password", key="delete_account_password")
                    if st.button("Delete My Account", use_container_width=True, type="secondary"):
                        if not confirm_delete:
                            st.warning("Please confirm that you understand this action is permanent.")
                        elif not delete_pwd:
                            st.warning("Please enter your password to continue.")
                        else:
                            ok, msg = delete_user_account_self(current_user.id, delete_pwd)
                            if ok:
                                st.success(msg)
                                logout()
                            else:
                                st.error(msg)

    if current_user is None:
        render_login_signup()
        render_guide_and_examples()
        st.markdown("---")
        st.caption("Disclaimer: This application is not a substitute for professional medical diagnosis.")
        return

    if current_user.role == "admin":
        page = st.radio("Navigation", ["Admin Dashboard", "Class Guide"], horizontal=True)
        if page == "Admin Dashboard":
            render_admin_page(current_user)
        else:
            render_guide_and_examples()
    else:
        page = st.radio("Navigation", ["Predict", "Class Guide"], horizontal=True)
        if page == "Predict":
            render_user_prediction_page(current_user)
        else:
            render_guide_and_examples()

    st.markdown("---")
    st.caption("Disclaimer: This application is not a substitute for professional medical diagnosis.")


if __name__ == "__main__":
    main()
