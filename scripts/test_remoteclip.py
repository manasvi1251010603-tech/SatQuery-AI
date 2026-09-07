from pathlib import Path

import torch
import open_clip
from PIL import Image


MODEL_PATH = Path(
    "models/remoteclip/RemoteCLIP-ViT-B-32.pt"
)

IMAGE_PATH = Path(
    "data/demo/bright/sample/"
    "marshall-wildfire_00000000_pre_disaster.tif"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found: {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    print("Loading RemoteCLIP ViT-B-32...")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            "ViT-B-32"
        )
    )

    tokenizer = open_clip.get_tokenizer(
        "ViT-B-32"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.eval()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = model.to(device)

    print(f"Device: {device}")
    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # TEXT CANDIDATES
    # --------------------------------------------------------

    texts = [
        "a residential area with buildings",
        "an airport with airplanes",
        "a dense urban area",
        "a forested area",
        "a rural landscape",
        "a wildfire affected area",
        "roads and buildings",
    ]

    text_tensor = tokenizer(
        texts
    ).to(device)

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    with torch.no_grad():

        image_features = model.encode_image(
            image_tensor
        )

        text_features = model.encode_text(
            text_tensor
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        similarity = (
            100.0
            * image_features
            @ text_features.T
        )

        probabilities = (
            similarity
            .softmax(dim=-1)
            .squeeze(0)
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = list(
        zip(
            texts,
            probabilities.cpu().tolist(),
        )
    )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("=" * 60)
    print("REMOTECLIP IMAGE-TEXT RETRIEVAL")
    print("=" * 60)

    for text, score in results:
        print(
            f"{score * 100:6.2f}%  {text}"
        )

    print("=" * 60)
    print(
        f"Top match: {results[0][0]}"
    )
    print(
        f"Top score: {results[0][1] * 100:.2f}%"
    )

    print()
    print("REMOTECLIP TEST PASSED")


if __name__ == "__main__":
    main()