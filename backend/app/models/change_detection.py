
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "deepang/adaptformer-LEVIR-CD"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

OUTPUT_DIR = PROJECT_ROOT / "data" / "demo" / "results"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASK_PATH = OUTPUT_DIR / "change_mask.png"
OVERLAY_PATH = OUTPUT_DIR / "change_overlay.png"
DIFF_PATH = OUTPUT_DIR / "change_difference.png"


# ============================================================
# DEVICE
# ============================================================
#
# AdaptFormer on your current setup was working on CPU.
# Keeping the change model on CPU avoids MPS-specific issues
# and does not affect the rest of SatQuery.
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# DETECTION PARAMETERS
# ============================================================

# Very small model artifacts are not useful as change regions.
MIN_REGION_AREA = 100

# Morphological filtering for fallback visual-difference analysis.
MORPH_KERNEL = 5

# If model prediction is below this ratio, we consider it
# effectively empty and run the independent visual-difference
# fallback.
MODEL_MIN_CHANGE_RATIO = 0.0005

# Difference fallback:
# percentile threshold is adaptive to each image pair.
DIFF_PERCENTILE = 92

# Minimum difference threshold.
DIFF_MIN_THRESHOLD = 22

# Remove tiny connected components.
DIFF_MIN_COMPONENT_AREA = 120

# Cap reported regions.
MAX_REGIONS = 10


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_change_model():

    print(
        f"[Change] Loading {MODEL_NAME}..."
    )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        "[Change] AdaptFormer loaded."
    )

    return processor, model, DEVICE


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_path: str | Path,
) -> Image.Image:

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
    # TIFF / GeoTIFF
    # --------------------------------------------------------

    if suffix in {".tif", ".tiff"}:

        try:

            import rasterio

            with rasterio.open(path) as src:

                if src.count >= 3:

                    bands = src.read(
                        [1, 2, 3]
                    ).astype(
                        np.float32
                    )

                    rgb = []

                    for band in bands:

                        finite = np.isfinite(
                            band
                        )

                        if not np.any(finite):

                            normalized = (
                                np.zeros_like(
                                    band,
                                    dtype=np.uint8,
                                )
                            )

                        else:

                            lo, hi = np.percentile(
                                band[finite],
                                [2, 98],
                            )

                            if hi <= lo:

                                normalized = (
                                    np.zeros_like(
                                        band,
                                        dtype=np.uint8,
                                    )
                                )

                            else:

                                normalized = (
                                    (
                                        band - lo
                                    )
                                    / (
                                        hi - lo
                                    )
                                    * 255.0
                                )

                                normalized = (
                                    np.clip(
                                        normalized,
                                        0,
                                        255,
                                    )
                                    .astype(
                                        np.uint8
                                    )
                                )

                        rgb.append(
                            normalized
                        )

                    array = np.stack(
                        rgb,
                        axis=-1,
                    )

                    return Image.fromarray(
                        array,
                        mode="RGB",
                    )

                if src.count == 1:

                    band = src.read(
                        1
                    ).astype(
                        np.float32
                    )

                    finite = np.isfinite(
                        band
                    )

                    if np.any(finite):

                        lo, hi = np.percentile(
                            band[finite],
                            [2, 98],
                        )

                        if hi > lo:

                            band = (
                                (
                                    band - lo
                                )
                                / (
                                    hi - lo
                                )
                                * 255
                            )

                        else:

                            band = np.zeros_like(
                                band
                            )

                    else:

                        band = np.zeros_like(
                            band
                        )

                    band = np.clip(
                        band,
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )

                    array = np.stack(
                        [
                            band,
                            band,
                            band,
                        ],
                        axis=-1,
                    )

                    return Image.fromarray(
                        array,
                        mode="RGB",
                    )

        except ImportError:
            pass

        except Exception as exc:

            print(
                "[Change] Rasterio TIFF read "
                f"failed: {exc}"
            )

    # --------------------------------------------------------
    # PNG/JPG/WEBP/BMP/etc.
    # --------------------------------------------------------

    return Image.open(
        path
    ).convert("RGB")


# ============================================================
# IMAGE ALIGNMENT
# ============================================================

def align_pair(
    before: Image.Image,
    after: Image.Image,
) -> tuple[Image.Image, Image.Image]:

    # For change detection, both images need identical
    # dimensions before preprocessing.
    if before.size == after.size:
        return before, after

    print(
        "[Change] Resizing temporal pair:"
        f" before={before.size}, after={after.size}"
    )

    after = after.resize(
        before.size,
        Image.Resampling.BILINEAR,
    )

    return before, after


# ============================================================
# MODEL OUTPUT -> BINARY MASK
# ============================================================

def model_logits_to_mask(
    logits: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, float]:

    """
    Converts AdaptFormer's output into:
        binary mask
        confidence

    Supports common [B,C,H,W] output.
    """

    if logits.ndim != 4:
        raise ValueError(
            f"Unexpected logits shape: {tuple(logits.shape)}"
        )

    # First sample.
    sample = logits[0]

    if sample.shape[0] >= 2:

        probabilities = torch.softmax(
            sample,
            dim=0,
        )

        change_probability = probabilities[1]

        confidence = float(
            change_probability.max()
            .detach()
            .cpu()
            .item()
        )

        raw_mask = (
            change_probability >= 0.5
        ).float()

    else:

        # Binary one-channel fallback.
        probability = torch.sigmoid(
            sample[0]
        )

        confidence = float(
            probability.max()
            .detach()
            .cpu()
            .item()
        )

        raw_mask = (
            probability >= 0.5
        ).float()

    mask = (
        raw_mask
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )

    width, height = target_size

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask, confidence


# ============================================================
# MASK CLEANING
# ============================================================

def clean_mask(
    mask: np.ndarray,
    minimum_area: int = MIN_REGION_AREA,
) -> np.ndarray:

    binary = (
        mask > 0
    ).astype(
        np.uint8
    ) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL,
            MORPH_KERNEL,
        ),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Connected-component filtering.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    for index in range(
        1,
        count,
    ):

        area = int(
            stats[
                index,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= minimum_area:

            cleaned[
                labels == index
            ] = 255

    return cleaned


# ============================================================
# INDEPENDENT VISUAL DIFFERENCE FALLBACK
# ============================================================

def compute_visual_difference(
    before: Image.Image,
    after: Image.Image,
) -> tuple[np.ndarray, float]:

    """
    Independent image-difference analysis.

    This is NOT claimed to be AdaptFormer.
    It exists so that a zero-output model does not force
    SatQuery to incorrectly say "nothing changed".
    """

    before_array = np.array(
        before
    ).astype(
        np.uint8
    )

    after_array = np.array(
        after
    ).astype(
        np.uint8
    )

    before_gray = cv2.cvtColor(
        before_array,
        cv2.COLOR_RGB2GRAY,
    )

    after_gray = cv2.cvtColor(
        after_array,
        cv2.COLOR_RGB2GRAY,
    )

    # Light blur reduces one-pixel sensor/compression noise.
    before_gray = cv2.GaussianBlur(
        before_gray,
        (5, 5),
        0,
    )

    after_gray = cv2.GaussianBlur(
        after_gray,
        (5, 5),
        0,
    )

    difference = cv2.absdiff(
        before_gray,
        after_gray,
    )

    # Adaptive threshold from image statistics.
    percentile_value = float(
        np.percentile(
            difference,
            DIFF_PERCENTILE,
        )
    )

    threshold_value = int(
        max(
            DIFF_MIN_THRESHOLD,
            percentile_value,
        )
    )

    _, mask = cv2.threshold(
        difference,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL,
            MORPH_KERNEL,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = clean_mask(
        mask,
        minimum_area=DIFF_MIN_COMPONENT_AREA,
    )

    change_ratio = float(
        np.count_nonzero(mask)
        / mask.size
        if mask.size
        else 0.0
    )

    return mask, change_ratio


# ============================================================
# REGIONS
# ============================================================

def extract_regions(
    binary_mask: np.ndarray,
) -> list[dict]:

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < MIN_REGION_AREA:
            continue

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area_pixels": round(
                    area,
                    2,
                ),
            }
        )

    regions.sort(
        key=lambda item: item[
            "area_pixels"
        ],
        reverse=True,
    )

    return regions[:MAX_REGIONS]


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    after: Image.Image,
    mask: np.ndarray,
) -> Image.Image:

    after_array = np.array(
        after
    ).copy()

    changed = mask > 0

    # Blend red instead of replacing the whole pixel.
    after_array[changed] = (
        0.45 * after_array[changed]
        + 0.55 * np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
    ).astype(
        np.uint8
    )

    # Add contours for clear evidence.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        after_array,
        contours,
        -1,
        (255, 0, 0),
        2,
    )

    return Image.fromarray(
        after_array,
        mode="RGB",
    )


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def question_type(
    question: str,
) -> str:

    q = question.lower()

    if any(
        word in q
        for word in [
            "where",
            "location",
            "locate",
            "which area",
            "which areas",
        ]
    ):
        return "where"

    if any(
        word in q
        for word in [
            "how much",
            "percentage",
            "percent",
            "extent",
            "area",
        ]
    ):
        return "extent"

    if any(
        word in q
        for word in [
            "when",
            "date",
            "temporal",
        ]
    ):
        return "temporal"

    return "what"


# ============================================================
# NATURAL LANGUAGE ANSWER
# ============================================================

def build_answer(
    question: str,
    regions: list[dict],
    change_percentage: float,
    method: str,
) -> str:

    qtype = question_type(
        question
    )

    if regions:

        if method == "AdaptFormer + visual fallback":

            basis = (
                "AdaptFormer did not produce a reliable "
                "change mask, so SatQuery additionally "
                "used independent visual-difference "
                "analysis to identify candidate changed "
                "regions."
            )

        elif method == "visual difference":

            basis = (
                "Candidate changes were identified using "
                "visual-difference analysis between the "
                "two images."
            )

        else:

            basis = (
                "The change-detection model identified "
                "the highlighted regions."
            )

        if qtype == "where":

            return (
                f"{basis} "
                f"Potential change is concentrated in "
                f"{len(regions)} highlighted region(s)."
            )

        if qtype == "extent":

            return (
                f"{basis} "
                f"Approximately "
                f"{change_percentage:.2f}% "
                f"of the analyzed pixels are "
                f"in the detected change mask."
            )

        return (
            f"{basis} "
            f"Approximately "
            f"{change_percentage:.2f}% "
            f"of the analyzed pixels are "
            f"classified as changed."
        )

    # No reliable candidate areas.
    return (
        "No reliable change regions were identified "
        "between the supplied images. This should be "
        "interpreted as low detected change, not proof "
        "that the two scenes are identical."
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_change_detection(
    before_path: str,
    after_path: str,
    question: str = "What changed?",
) -> dict:

    before_file = Path(
        before_path
    )

    after_file = Path(
        after_path
    )

    if not before_file.exists():

        raise FileNotFoundError(
            f"Before image not found: "
            f"{before_file}"
        )

    if not after_file.exists():

        raise FileNotFoundError(
            f"After image not found: "
            f"{after_file}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    before = load_image(
        before_file
    )

    after = load_image(
        after_file
    )

    # --------------------------------------------------------
    # Align image pair
    # --------------------------------------------------------

    before, after = align_pair(
        before,
        after,
    )

    target_size = after.size

    print(
        "[Change] Before:",
        before.size,
    )

    print(
        "[Change] After:",
        after.size,
    )

    # --------------------------------------------------------
    # AdaptFormer inference
    # --------------------------------------------------------

    model_mask = np.zeros(
        (
            target_size[1],
            target_size[0],
        ),
        dtype=np.uint8,
    )

    model_confidence = 0.0

    model_change_ratio = 0.0

    model_error = None

    try:

        processor, model, device = (
            load_change_model()
        )

        inputs = processor(
            images=(before, after),
            return_tensors="pt",
        )

        moved_inputs = {}

        for key, value in inputs.items():

            if hasattr(value, "to"):

                moved_inputs[key] = (
                    value.to(device)
                )

            else:

                moved_inputs[key] = value

        with torch.inference_mode():

            outputs = model(
                **moved_inputs
            )

        if not hasattr(
            outputs,
            "logits",
        ):

            raise ValueError(
                "AdaptFormer output does not "
                "contain logits."
            )

        model_mask, model_confidence = (
            model_logits_to_mask(
                outputs.logits,
                target_size,
            )
        )

        model_mask = clean_mask(
            model_mask,
            minimum_area=MIN_REGION_AREA,
        )

        model_change_ratio = float(
            np.count_nonzero(model_mask)
            / model_mask.size
            if model_mask.size
            else 0.0
        )

        print(
            "[Change] AdaptFormer "
            f"change ratio: "
            f"{model_change_ratio * 100:.4f}%"
        )

        print(
            "[Change] AdaptFormer "
            f"confidence: "
            f"{model_confidence:.4f}"
        )

    except Exception as exc:

        model_error = str(exc)

        print(
            "[Change] AdaptFormer failed:",
            exc,
        )

    # --------------------------------------------------------
    # Decide whether model result is usable
    # --------------------------------------------------------

    use_model = (
        model_error is None
        and model_change_ratio
        >= MODEL_MIN_CHANGE_RATIO
    )

    # --------------------------------------------------------
    # Fallback visual-difference analysis
    # --------------------------------------------------------

    fallback_mask = None

    fallback_ratio = 0.0

    if not use_model:

        print(
            "[Change] AdaptFormer mask is empty "
            "or unreliable."
        )

        print(
            "[Change] Running visual-difference fallback."
        )

        fallback_mask, fallback_ratio = (
            compute_visual_difference(
                before,
                after,
            )
        )

        print(
            "[Change] Visual-difference "
            f"ratio: {fallback_ratio * 100:.4f}%"
        )

    # --------------------------------------------------------
    # Select final mask
    # --------------------------------------------------------

    if use_model:

        final_mask = model_mask

        method = "AdaptFormer"

        confidence = model_confidence

    elif fallback_mask is not None:

        final_mask = fallback_mask

        method = "visual difference"

        # This is deliberately not presented as
        # AdaptFormer confidence.
        confidence = min(
            0.99,
            max(
                0.05,
                fallback_ratio * 5.0,
            ),
        )

    else:

        final_mask = model_mask

        method = "AdaptFormer + visual fallback"

        confidence = model_confidence

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    final_mask = clean_mask(
        final_mask,
        minimum_area=MIN_REGION_AREA,
    )

    changed_pixels = int(
        np.count_nonzero(
            final_mask
        )
    )

    total_pixels = int(
        final_mask.size
    )

    change_ratio = (
        changed_pixels / total_pixels
        if total_pixels
        else 0.0
    )

    change_percentage = (
        change_ratio * 100.0
    )

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    mask_image = Image.fromarray(
        final_mask,
        mode="L",
    )

    mask_image.save(
        MASK_PATH
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        after,
        final_mask,
    )

    overlay.save(
        OVERLAY_PATH
    )

    # --------------------------------------------------------
    # Save grayscale visual difference
    # --------------------------------------------------------

    before_np = np.array(
        before
    )

    after_np = np.array(
        after
    )

    before_gray = cv2.cvtColor(
        before_np,
        cv2.COLOR_RGB2GRAY,
    )

    after_gray = cv2.cvtColor(
        after_np,
        cv2.COLOR_RGB2GRAY,
    )

    diff = cv2.absdiff(
        before_gray,
        after_gray,
    )

    diff = cv2.normalize(
        diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        diff,
        mode="L",
    ).save(
        DIFF_PATH
    )

    # --------------------------------------------------------
    # Regions
    # --------------------------------------------------------

    regions = extract_regions(
        final_mask
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer = build_answer(
        question=question,
        regions=regions,
        change_percentage=change_percentage,
        method=method,
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "answer": answer,

        "confidence": round(
            float(confidence),
            4,
        ),

        "model_confidence": round(
            float(model_confidence),
            4,
        ),

        "change_percentage": round(
            change_percentage,
            4,
        ),

        "change_mask": str(
            MASK_PATH
        ),

        "change_overlay": str(
            OVERLAY_PATH
        ),

        "change_difference": str(
            DIFF_PATH
        ),

        "regions": regions,

        "model": MODEL_NAME,

        "method": method,

        "question": question,

        "image_size": {
            "width": target_size[0],
            "height": target_size[1],
        },

        "model_change_percentage": round(
            model_change_ratio * 100.0,
            4,
        ),

        "fallback_change_percentage": round(
            fallback_ratio * 100.0,
            4,
        ),

        "model_error": model_error,
    }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SatQuery multi-temporal "
            "change detection"
        )
    )

    parser.add_argument(
        "before",
        help="Path to before image",
    )

    parser.add_argument(
        "after",
        help="Path to after image",
    )

    parser.add_argument(
        "--question",
        default="What changed?",
        help="Natural-language question",
    )

    args = parser.parse_args()

    result = run_change_detection(
        before_path=args.before,
        after_path=args.after,
        question=args.question,
    )

    print()
    print(
        "========================================"
    )
    print(
        "SATQUERY CHANGE DETECTION RESULT"
    )
    print(
        "========================================"
    )

    print(
        "Answer:",
        result["answer"],
    )

    print(
        "Method:",
        result["method"],
    )

    print(
        "Confidence:",
        result["confidence"],
    )

    print(
        "Model confidence:",
        result["model_confidence"],
    )

    print(
        "Change:",
        f"{result['change_percentage']:.2f}%",
    )

    print(
        "Model change:",
        f"{result['model_change_percentage']:.2f}%",
    )

    print(
        "Fallback change:",
        f"{result['fallback_change_percentage']:.2f}%",
    )

    print(
        "Regions:",
        len(result["regions"]),
    )

    print(
        "Mask:",
        result["change_mask"],
    )

    print(
        "Overlay:",
        result["change_overlay"],
    )

    print(
        "Difference:",
        result["change_difference"],
    )

    print()

    for region in result["regions"]:
        print(region)
