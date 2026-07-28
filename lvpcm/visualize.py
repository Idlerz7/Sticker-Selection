"""Compact paginated neighbor contact sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

from PIL import Image, ImageDraw, ImageOps

from .images import project_rgb

_RESAMPLING = getattr(Image, "Resampling", Image)


def _tile(path: Path, label: str, size: int = 112) -> Image.Image:
    data = path.read_bytes()
    image = project_rgb(data)
    image.thumbnail((size, size), _RESAMPLING.LANCZOS)
    canvas = Image.new("RGB", (size, size + 18), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    ImageDraw.Draw(canvas).text((2, size + 2), label, fill="black")
    return ImageOps.expand(canvas, border=1, fill="gray")


def contact_sheet_pages(
    anchors: Sequence[int], neighbors: Dict[int, Sequence[int]], id2path: Dict[int, Path],
    output_dir: Path, prefix: str, rows_per_page: int = 20,
) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for page_start in range(0, len(anchors), rows_per_page):
        page = anchors[page_start:page_start + rows_per_page]
        width, height = 6 * 114, len(page) * 132
        canvas = Image.new("RGB", (width, height), "white")
        for row_index, anchor in enumerate(page):
            ids = [int(anchor)] + [int(value) for value in neighbors[int(anchor)][:5]]
            for column, image_id in enumerate(ids):
                label = ("A:" if column == 0 else "N%d:" % column) + str(image_id)
                canvas.paste(_tile(id2path[image_id], label), (column * 114, row_index * 132))
        path = output_dir / ("%s_page_%03d.jpg" % (prefix, page_start // rows_per_page + 1))
        canvas.save(path, quality=90)
        paths.append(str(path))
    return paths
