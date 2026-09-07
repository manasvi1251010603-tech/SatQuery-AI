from pathlib import Path

import torch
from transformers import pipeline


IMAGE_PATH = Path("data/demo/sentinel2_rgb.png")


def main() -> None:
    print("PyTorch version:", torch.__version__)

    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU")

    classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
        device=device,
    )

    print("\nRunning inference...")

    results = classifier(str(IMAGE_PATH))

    print("\n========== MODEL OUTPUT ==========")

    for result in results[:5]:
        print(
            f"{result['label']}: "
            f"{result['score']:.4f}"
        )


if __name__ == "__main__":
    main()