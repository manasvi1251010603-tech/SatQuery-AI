from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "deepang/adaptformer-LEVIR-CD"

BEFORE_IMAGE = Path(
    "data/demo/adaptformer/before.png"
)

AFTER_IMAGE = Path(
    "data/demo/adaptformer/after.png"
)

OUTPUT_PATH = Path(
    "data/demo/adaptformer/change_mask.png"
)


def main() -> None:
    print("Loading processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading model...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # MPS is not always supported by every custom model operation.
    # Start on CPU for maximum compatibility.
    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    before = Image.open(BEFORE_IMAGE).convert("RGB")
    after = Image.open(AFTER_IMAGE).convert("RGB")

    print("Before:", before.size)
    print("After :", after.size)

    inputs = processor(
        images=(before, after),
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("Running change detection...")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    print("Logits shape:", logits.shape)

    prediction = logits.argmax(dim=1)[0]

    mask = prediction.cpu().numpy().astype(np.uint8)

    print("Unique prediction values:", np.unique(mask))

    # Convert prediction to visible binary mask.
    visible_mask = (mask > 0).astype(np.uint8) * 255

    output = Image.fromarray(visible_mask)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.save(OUTPUT_PATH)

    print("Saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()