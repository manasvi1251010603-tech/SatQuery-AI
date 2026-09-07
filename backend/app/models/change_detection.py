from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "deepang/adaptformer-LEVIR-CD"


@lru_cache(maxsize=1)
def load_change_model():
    print("Loading AdaptFormer...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print("AdaptFormer loaded.")

    return processor, model, device


def run_change_detection(
    before_path: str,
    after_path: str,
    question: str = "What changed?",
) -> dict:

    before_file = Path(before_path)
    after_file = Path(after_path)

    if not before_file.exists():
        raise FileNotFoundError(
            f"Before image not found: {before_file}"
        )

    if not after_file.exists():
        raise FileNotFoundError(
            f"After image not found: {after_file}"
        )

    processor, model, device = load_change_model()

    before = Image.open(
        before_file
    ).convert("RGB")

    after = Image.open(
        after_file
    ).convert("RGB")

    inputs = processor(
        images=(before, after),
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    prediction = logits.argmax(
        dim=1
    )[0]

    mask = prediction.cpu().numpy().astype(
        np.uint8
    )

    changed_pixels = int(
        np.count_nonzero(mask)
    )

    total_pixels = int(mask.size)

    change_ratio = (
        changed_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    # --------------------------------
    # Full-resolution mask
    # --------------------------------

    visible_mask = (
        (mask > 0).astype(np.uint8) * 255
    )

    mask_image = Image.fromarray(
        visible_mask
    )

    mask_image = mask_image.resize(
        after.size,
        Image.Resampling.NEAREST,
    )

    output_dir = Path(
        "data/demo/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        output_dir / "change_mask.png"
    )

    overlay_path = (
        output_dir / "change_overlay.png"
    )

    mask_image.save(mask_path)

    # --------------------------------
    # Overlay
    # --------------------------------

    after_array = np.array(after).copy()

    mask_array = np.array(
        mask_image
    )

    changed = mask_array > 0

    after_array[changed] = [
        255,
        0,
        0,
    ]

    overlay = Image.fromarray(
        after_array
    )

    overlay.save(overlay_path)

    # --------------------------------
    # Changed regions
    # --------------------------------

    binary_mask = (
        (mask_array > 0)
        .astype(np.uint8)
        * 255
    )

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 100:
            continue

        x, y, width, height = (
            cv2.boundingRect(contour)
        )

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area_pixels": float(area),
            }
        )

    regions.sort(
        key=lambda r: r["area_pixels"],
        reverse=True,
    )

    regions = regions[:10]

    # --------------------------------
    # Basic natural-language response
    # --------------------------------

    query = question.lower()

    if any(
        word in query
        for word in [
            "where",
            "location",
            "locate",
        ]
    ):

        if regions:
            answer = (
                "Potential change was detected "
                "in the highlighted regions."
            )
        else:
            answer = (
                "No significant change regions "
                "were detected."
            )

    elif any(
        word in query
        for word in [
            "how much",
            "percentage",
            "percent",
            "extent",
        ]
    ):

        answer = (
            f"Approximately "
            f"{change_ratio * 100:.2f}% "
            f"of the analyzed pixels were "
            f"classified as changed."
        )

    else:

        if changed_pixels > 0:

            answer = (
                "Potential change was detected "
                f"in {change_ratio * 100:.2f}% "
                "of the analyzed pixels."
            )

        else:

            answer = (
                "No change was detected by "
                "the change-detection model."
            )

    return {
        "answer": answer,
        "confidence": None,
        "change_percentage": (
            change_ratio * 100
        ),
        "change_mask": str(mask_path),
        "change_overlay": str(overlay_path),
        "regions": regions,
        "model": MODEL_NAME,
        "question": question,
    }