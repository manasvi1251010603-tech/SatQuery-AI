from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)


MODEL_NAME = "IDEA-Research/grounding-dino-tiny"


@lru_cache(maxsize=1)
def load_grounding_model():
    print("Loading Grounding DINO...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    print("Grounding DINO loaded.")
    print("Device:", device)

    return processor, model, device


def run_grounding(
    image_path: str,
    query: str,
) -> dict:

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_file}"
        )

    processor, model, device = (
        load_grounding_model()
    )

    image = Image.open(
        image_file
    ).convert("RGB")

    # Grounding DINO works best when text
    # concepts are separated by periods.
    text_query = query.strip()

    if not text_query.endswith("."):
        text_query += "."

    inputs = processor(
        images=image,
        text=text_query,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.25,
        text_threshold=0.20,
        target_sizes=[
            image.size[::-1]
        ],
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["text_labels"]

    annotated = image.copy()

    draw = ImageDraw.Draw(
        annotated
    )

    detections = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):

        x0, y0, x1, y1 = [
            int(value)
            for value in box
        ]

        draw.rectangle(
            [x0, y0, x1, y1],
            outline="red",
            width=4,
        )

        draw.text(
            (x0, max(0, y0 - 15)),
            f"{label} {score:.2f}",
            fill="red",
        )

        detections.append(
            {
                "label": label,
                "confidence": float(score),
                "box": [
                    x0,
                    y0,
                    x1,
                    y1,
                ],
            }
        )

    output_dir = Path(
        "data/demo/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "grounding_result.png"
    )

    annotated.save(
        output_path
    )

    return {
        "answer": (
            f"Detected {len(detections)} "
            f"candidate region(s) for "
            f"'{query}'."
        ),
        "confidence": (
            max(
                (
                    d["confidence"]
                    for d in detections
                ),
                default=0.0,
            )
        ),
        "model": MODEL_NAME,
        "query": query,
        "detections": detections,
        "grounding_image": str(
            output_path
        ),
    }