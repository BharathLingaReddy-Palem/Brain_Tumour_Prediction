"""Consistent, bounded image decoding for all upload entry points."""
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def decode_image(source, mode="RGB"):
    """Fully decode bytes, paths or UploadedFile objects; return a detached PIL image."""
    if hasattr(source, "getvalue"):
        source = source.getvalue()
    if isinstance(source, (bytes, bytearray)):
        if not source or len(source) > MAX_UPLOAD_BYTES:
            raise ValueError("Image must be non-empty and no larger than 32 MB.")
        source = BytesIO(source)
    elif isinstance(source, (str, Path)) and Path(source).stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Image must be no larger than 32 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel limit.")
                image.load()
                return ImageOps.exif_transpose(image).convert(mode)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Cannot read this image. Upload a valid, undamaged MRI image file.") from exc
