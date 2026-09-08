
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    import rasterio
except Exception:
    rasterio = None


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "data" / "demo" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GROUNDING_OUTPUT = RESULTS_DIR / "grounding_result.png"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ============================================================
# GROUNDING SETTINGS
# ============================================================
#
# We intentionally use lower thresholds than the official
# example because remote-sensing objects are small and dense.
#
PRIMARY_THRESHOLD = 0.16
PRIMARY_TEXT_THRESHOLD = 0.08

FALLBACK_THRESHOLD = 0.10
FALLBACK_TEXT_THRESHOLD = 0.05

NMS_IOU = 0.35

# Tiling is the important part for small aerial/satellite
# objects. A 256x256 image is still analysed correctly.
TILE_SIZE = 512
TILE_OVERLAP = 128

MAX_DETECTIONS = 60

MIN_BOX_WIDTH = 4
MIN_BOX_HEIGHT = 4

# Ignore boxes that cover nearly the entire tile/image.
MAX_BOX_AREA_FRACTION = 0.75


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_grounding_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_NAME
    )

    model = model.to(DEVICE)
    model.eval()

    print(f"[Grounding] Model: {MODEL_NAME}")
    print(f"[Grounding] Device: {DEVICE}")

    return processor, model


# ============================================================
# IMAGE LOADING
# ============================================================

def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)

    finite = np.isfinite(band)

    if not np.any(finite):
        return np.zeros_like(band, dtype=np.uint8)

    lo, hi = np.percentile(
        band[finite],
        [2, 98],
    )

    if hi <= lo:
        return np.zeros_like(band, dtype=np.uint8)

    band = (band - lo) / (hi - lo)
    band = np.clip(band * 255.0, 0, 255)

    return band.astype(np.uint8)


def load_satellite_image(
    image_path: str | Path,
) -> Image.Image:
    """
    Accepts common image files and GeoTIFF/TIFF.

    Supported:
        PNG
        JPG/JPEG
        WEBP
        BMP
        TIFF/GeoTIFF
        other formats readable by Pillow
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Image is empty: {path}"
        )

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # GeoTIFF/TIFF
    # --------------------------------------------------------

    if suffix in {".tif", ".tiff"} and rasterio is not None:
        try:
            with rasterio.open(path) as src:

                if src.count >= 3:
                    bands = src.read([1, 2, 3])

                    rgb = np.stack(
                        [
                            _normalize_band(bands[0]),
                            _normalize_band(bands[1]),
                            _normalize_band(bands[2]),
                        ],
                        axis=-1,
                    )

                    return Image.fromarray(
                        rgb,
                        mode="RGB",
                    )

                if src.count == 1:
                    band = _normalize_band(
                        src.read(1)
                    )

                    rgb = np.stack(
                        [band, band, band],
                        axis=-1,
                    )

                    return Image.fromarray(
                        rgb,
                        mode="RGB",
                    )

        except Exception as exc:
            print(
                "[Grounding] Rasterio read failed; "
                f"using Pillow fallback: {exc}"
            )

    # --------------------------------------------------------
    # Normal image formats
    # --------------------------------------------------------

    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Could not decode image '{path}': {exc}"
        ) from exc


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:
    q = query.strip().lower()

    replacements = {
        "buildings": "building",
        "houses": "house",
        "homes": "house",
        "roads": "road",
        "streets": "street",
        "vehicles": "vehicle",
        "cars": "car",
        "trees": "tree",
        "water bodies": "water",
        "waterbody": "water",
    }

    return replacements.get(q, q)


def query_variants(query: str) -> list[str]:
    """
    Multiple prompts improve recall while keeping the
    user-facing result as the requested class.
    """

    q = normalize_query(query)

    mapping = {
        "building": [
            "building",
            "house",
            "residential building",
            "a building",
            "a house",
        ],
        "house": [
            "house",
            "building",
            "residential building",
            "a house",
            "a building",
        ],
        "road": [
            "road",
            "street",
            "a road",
            "a street",
        ],
        "street": [
            "street",
            "road",
            "a street",
            "a road",
        ],
        "car": [
            "car",
            "vehicle",
            "a car",
            "a vehicle",
        ],
        "vehicle": [
            "vehicle",
            "car",
            "a vehicle",
            "a car",
        ],
        "tree": [
            "tree",
            "a tree",
            "vegetation",
        ],
        "vegetation": [
            "vegetation",
            "tree",
            "a tree",
        ],
        "water": [
            "water",
            "a water body",
            "water body",
        ],
    }

    return mapping.get(q, [q, f"a {q}"])


# ============================================================
# TILE GENERATION
# ============================================================

def _positions(total: int, tile: int, overlap: int) -> list[int]:
    if total <= tile:
        return [0]

    step = max(1, tile - overlap)
    values = list(range(0, max(1, total - tile + 1), step))

    final_start = total - tile

    if values[-1] != final_start:
        values.append(final_start)

    return sorted(set(values))


def make_tiles(
    image: Image.Image,
) -> list[tuple[Image.Image, int, int]]:
    """
    Returns:
        tile_image, left, top
    """

    width, height = image.size

    xs = _positions(
        width,
        TILE_SIZE,
        TILE_OVERLAP,
    )

    ys = _positions(
        height,
        TILE_SIZE,
        TILE_OVERLAP,
    )

    tiles = []

    for top in ys:
        for left in xs:

            right = min(
                left + TILE_SIZE,
                width,
            )

            bottom = min(
                top + TILE_SIZE,
                height,
            )

            tile = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

            tiles.append(
                (
                    tile,
                    left,
                    top,
                )
            )

    return tiles


# ============================================================
# POST-PROCESSING COMPATIBILITY
# ============================================================

def post_process(
    processor,
    outputs,
    input_ids,
    image_size: tuple[int, int],
    threshold: float,
    text_threshold: float,
):
    """
    Supports both the older and newer Transformers APIs.

    Current HF examples use `threshold=...`; older examples
    used `box_threshold=...`.
    """

    target_sizes = [
        image_size[::-1]
    ]

    try:
        # Current Transformers API.
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

    except TypeError:
        # Backward compatibility.
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )


# ============================================================
# DETECTION FOR ONE TILE
# ============================================================

def detect_tile(
    image: Image.Image,
    query: str,
    threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:

    processor, model = load_grounding_model()

    prompt = query.strip().lower()

    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    # Move tensors to selected device.
    inputs = inputs.to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = post_process(
        processor=processor,
        outputs=outputs,
        input_ids=inputs.input_ids,
        image_size=image.size,
        threshold=threshold,
        text_threshold=text_threshold,
    )

    if not results:
        return []

    result = results[0]

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    detections = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):

        coords = (
            box
            .detach()
            .cpu()
            .tolist()
        )

        confidence = float(
            score
            .detach()
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = coords

        width = x2 - x1
        height = y2 - y1

        if width < MIN_BOX_WIDTH:
            continue

        if height < MIN_BOX_HEIGHT:
            continue

        area = max(
            0.0,
            width * height,
        )

        tile_area = (
            image.width
            * image.height
        )

        if tile_area > 0:
            fraction = area / tile_area

            if fraction > MAX_BOX_AREA_FRACTION:
                continue

        detections.append(
            {
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "confidence": confidence,
                "raw_label": str(label),
                "query": query,
            }
        )

    return detections


# ============================================================
# MAP TILE BOX -> FULL IMAGE
# ============================================================

def translate_box(
    box: list[float],
    left: int,
    top: int,
    scale_x: float,
    scale_y: float,
) -> list[float]:

    x1, y1, x2, y2 = box

    return [
        (x1 / scale_x) + left,
        (y1 / scale_y) + top,
        (x2 / scale_x) + left,
        (y2 / scale_y) + top,
    ]


# ============================================================
# IOU
# ============================================================

def iou(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# NMS
# ============================================================

def nms(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda x: float(
            x["confidence"]
        ),
        reverse=True,
    )

    kept = []

    for current in ordered:

        is_duplicate = False

        for previous in kept:

            if (
                iou(
                    current["box"],
                    previous["box"],
                )
                >= NMS_IOU
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(current)

    return kept


# ============================================================
# USER-FACING LABEL
# ============================================================

def display_label(
    query: str,
) -> str:

    q = normalize_query(query)

    if q in {"building", "house"}:
        return "building"

    if q in {"road", "street"}:
        return "road"

    if q in {"car", "vehicle"}:
        return "vehicle"

    if q in {"tree", "vegetation"}:
        return "vegetation"

    if q in {"water", "water body"}:
        return "water"

    return q


# ============================================================
# DRAW
# ============================================================

def annotate_image(
    image: Image.Image,
    detections: list[dict[str, Any]],
    output: Path,
) -> None:

    result = image.convert("RGB").copy()

    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            13,
        )
    except Exception:
        font = ImageFont.load_default()

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )

        label = detection["label"]

        confidence = float(
            detection["confidence"]
        )

        text = (
            f"{label} "
            f"{confidence * 100:.1f}%"
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=(255, 0, 0),
            width=2,
        )

        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = (
            text_bbox[2]
            - text_bbox[0]
        )

        text_height = (
            text_bbox[3]
            - text_bbox[1]
        )

        label_y = max(
            0,
            y1 - text_height - 6,
        )

        draw.rectangle(
            [
                x1,
                label_y,
                x1 + text_width + 8,
                label_y + text_height + 6,
            ],
            fill=(0, 0, 0),
        )

        draw.text(
            (
                x1 + 4,
                label_y + 3,
            ),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    result.save(output)


# ============================================================
# MAIN GROUNDING PIPELINE
# ============================================================

def run_grounding(
    image_path: str,
    query: str,
) -> dict[str, Any]:

    image = load_satellite_image(
        image_path
    )

    original_width, original_height = (
        image.size
    )

    semantic_label = display_label(query)

    variants = query_variants(query)

    print(
        "[Grounding]"
        f" Image: {original_width}x{original_height}"
    )

    print(
        "[Grounding]"
        f" Query: {query}"
    )

    print(
        "[Grounding]"
        f" Variants: {variants}"
    )

    tiles = make_tiles(image)

    print(
        f"[Grounding] Tiles: {len(tiles)}"
    )

    detections: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Primary pass
    # --------------------------------------------------------

    for tile_index, (tile, left, top) in enumerate(
        tiles,
        start=1,
    ):

        tile_width, tile_height = (
            tile.size
        )

        # The Grounding DINO processor already performs
        # its own resize. We deliberately preserve the full
        # tile so small buildings receive more pixels.
        for variant in variants:

            try:

                found = detect_tile(
                    tile,
                    variant,
                    PRIMARY_THRESHOLD,
                    PRIMARY_TEXT_THRESHOLD,
                )

            except Exception as exc:

                print(
                    "[Grounding] Tile "
                    f"{tile_index}, query '{variant}' "
                    f"failed: {exc}"
                )

                continue

            # Map boxes from tile coordinates back to image.
            #
            # Since detect_tile returns coordinates in the
            # tile's original target size, no scaling is needed
            # here.
            for item in found:

                x1, y1, x2, y2 = (
                    item["box"]
                )

                translated = [
                    x1 + left,
                    y1 + top,
                    x2 + left,
                    y2 + top,
                ]

                item["box"] = translated
                item["label"] = semantic_label
                item["tile"] = tile_index

                # Keep only boxes inside full image.
                item["box"] = [
                    max(
                        0.0,
                        min(
                            float(original_width),
                            item["box"][0],
                        ),
                    ),
                    max(
                        0.0,
                        min(
                            float(original_height),
                            item["box"][1],
                        ),
                    ),
                    max(
                        0.0,
                        min(
                            float(original_width),
                            item["box"][2],
                        ),
                    ),
                    max(
                        0.0,
                        min(
                            float(original_height),
                            item["box"][3],
                        ),
                    ),
                ]

                detections.append(item)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not detections:

        print(
            "[Grounding] No primary detections."
        )

        print(
            "[Grounding] Running fallback pass."
        )

        for tile_index, (tile, left, top) in enumerate(
            tiles,
            start=1,
        ):

            for variant in variants[:2]:

                try:

                    found = detect_tile(
                        tile,
                        variant,
                        FALLBACK_THRESHOLD,
                        FALLBACK_TEXT_THRESHOLD,
                    )

                except Exception as exc:

                    print(
                        "[Grounding] Fallback tile "
                        f"{tile_index} failed: {exc}"
                    )

                    continue

                for item in found:

                    x1, y1, x2, y2 = (
                        item["box"]
                    )

                    item["box"] = [
                        x1 + left,
                        y1 + top,
                        x2 + left,
                        y2 + top,
                    ]

                    item["label"] = (
                        semantic_label
                    )

                    item["tile"] = tile_index

                    detections.append(
                        item
                    )

    # --------------------------------------------------------
    # Remove impossible boxes
    # --------------------------------------------------------

    valid = []

    for item in detections:

        x1, y1, x2, y2 = (
            item["box"]
        )

        width = x2 - x1
        height = y2 - y1

        if width < MIN_BOX_WIDTH:
            continue

        if height < MIN_BOX_HEIGHT:
            continue

        valid.append(item)

    detections = valid

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    detections = nms(
        detections
    )

    # --------------------------------------------------------
    # Limit count
    # --------------------------------------------------------

    detections = sorted(
        detections,
        key=lambda x: float(
            x["confidence"]
        ),
        reverse=True,
    )

    detections = detections[
        :MAX_DETECTIONS
    ]

    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    evidence = []

    for item in detections:

        x1, y1, x2, y2 = (
            item["box"]
        )

        evidence.append(
            {
                "label": semantic_label,
                "confidence": round(
                    float(
                        item["confidence"]
                    ),
                    4,
                ),
                "box": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ],
            }
        )

    # --------------------------------------------------------
    # Confidence / answer
    # --------------------------------------------------------

    if evidence:

        confidence = max(
            d["confidence"]
            for d in evidence
        )

        mean_confidence = float(
            np.mean(
                [
                    d["confidence"]
                    for d in evidence
                ]
            )
        )

        answer = (
            f"Detected {len(evidence)} "
            f"{semantic_label} region(s)."
        )

    else:

        confidence = 0.0
        mean_confidence = 0.0

        answer = (
            f"No reliable {semantic_label} "
            f"detections were found."
        )

    # --------------------------------------------------------
    # Save annotated result
    # --------------------------------------------------------

    annotate_image(
        image=image,
        detections=evidence,
        output=GROUNDING_OUTPUT,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "answer": answer,

        "confidence": round(
            float(confidence),
            4,
        ),

        "mean_confidence": round(
            float(mean_confidence),
            4,
        ),

        "model": MODEL_NAME,

        "device": DEVICE,

        "query": query,

        "normalized_query": normalize_query(
            query
        ),

        "detections": evidence,

        "grounding_image": str(
            GROUNDING_OUTPUT
        ),

        "image_size": {
            "width": original_width,
            "height": original_height,
        },

        "tile_count": len(tiles),

        "thresholds": {
            "primary": PRIMARY_THRESHOLD,
            "primary_text": PRIMARY_TEXT_THRESHOLD,
            "fallback": FALLBACK_THRESHOLD,
            "fallback_text": FALLBACK_TEXT_THRESHOLD,
        },
    }


# ============================================================
# DIRECT TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="SatQuery Grounding DINO test"
    )

    parser.add_argument(
        "image",
        help="Path to image",
    )

    parser.add_argument(
        "query",
        help="Text query, e.g. 'buildings'",
    )

    args = parser.parse_args()

    result = run_grounding(
        args.image,
        args.query,
    )

    print()
    print(
        "======================================"
    )

    print(
        "SATQUERY GROUNDING RESULT"
    )

    print(
        "======================================"
    )

    print(
        "Answer:",
        result["answer"],
    )

    print(
        "Confidence:",
        result["confidence"],
    )

    print(
        "Mean confidence:",
        result["mean_confidence"],
    )

    print(
        "Detections:",
        len(result["detections"]),
    )

    print(
        "Image:",
        result["image_size"],
    )

    print(
        "Tiles:",
        result["tile_count"],
    )

    print(
        "Output:",
        result["grounding_image"],
    )

    print()

    for detection in result["detections"]:
        print(detection)
