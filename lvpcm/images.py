"""Project-compatible first-frame image loading and frozen perturbations."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from img_utils import get_gif_nframe, judge_img_type, read_pil_image, remove_transparency

_RESAMPLING = getattr(Image, "Resampling", Image)


def first_frame(data: bytes) -> Image.Image:
    image = get_gif_nframe(data, 0) if judge_img_type(data) == "gif" else read_pil_image(data)
    image.load()
    return image


def project_rgb(data: bytes) -> Image.Image:
    return remove_transparency(first_frame(data), bg_colour=(255, 255, 255)).convert("RGB")


def alpha_bbox_crop(data: bytes) -> Tuple[Image.Image, bool]:
    source = first_frame(data)
    rgba = source.convert("RGBA")
    alpha = rgba.getchannel("A")
    extrema = alpha.getextrema()
    bbox = alpha.getbbox()
    width, height = rgba.size
    eligible = bool(
        extrema[0] < 255 and bbox and bbox != (0, 0, width, height)
        and (bbox[2] - bbox[0]) >= 8 and (bbox[3] - bbox[1]) >= 8
        and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.25 * width * height
    )
    if not eligible:
        return remove_transparency(source, bg_colour=(255, 255, 255)).convert("RGB"), False
    cropped = rgba.crop(bbox)
    return remove_transparency(cropped, bg_colour=(255, 255, 255)).convert("RGB"), True


def perturb_image(data: bytes, perturbation: str) -> Tuple[Image.Image, bool]:
    if perturbation == "alpha_bbox":
        return alpha_bbox_crop(data)
    clean = project_rgb(data)
    if perturbation == "clean":
        return clean, True
    if perturbation == "resize75":
        width, height = clean.size
        down = (max(1, round(width * 0.75)), max(1, round(height * 0.75)))
        return clean.resize(down, _RESAMPLING.BICUBIC).resize(clean.size, _RESAMPLING.BICUBIC), True
    if perturbation == "jpeg75":
        buffer = BytesIO()
        clean.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        value = Image.open(buffer).convert("RGB")
        value.load()
        return value, True
    raise ValueError("unknown perturbation: %s" % perturbation)


def canonical_rgb_hash(data: bytes) -> str:
    image = project_rgb(data)
    digest = hashlib.sha256()
    digest.update(("%dx%d:" % image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()
