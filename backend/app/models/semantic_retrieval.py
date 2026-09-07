from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import open_clip
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "remoteclip"
    / "RemoteCLIP-ViT-B-32.pt"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "retrieval_index"
)

EMBEDDINGS_PATH = (
    INDEX_DIR
    / "embeddings.npy"
)

METADATA_PATH = (
    INDEX_DIR
    / "metadata.npy"
)

MODEL_NAME = "RemoteCLIP ViT-B-32"


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MODEL STATE
# ============================================================

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = None


# ============================================================
# LOAD REMOTECLIP
# ============================================================

def load_remoteclip():
    """
    Load the verified RemoteCLIP ViT-B-32 checkpoint.

    This follows the exact loading path already verified by
    scripts/test_remoteclip.py.

    FAISS is intentionally not used here because its native
    OpenMP runtime conflicts with the current Apple Silicon
    PyTorch/OpenCLIP process.
    """

    global _MODEL
    global _PREPROCESS
    global _TOKENIZER
    global _DEVICE

    if _MODEL is not None:
        return (
            _MODEL,
            _PREPROCESS,
            _TOKENIZER,
            _DEVICE,
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not found: {MODEL_PATH}"
        )

    print(
        "Loading RemoteCLIP ViT-B-32..."
    )

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

    device = get_device()

    model = model.to(device)
    model.eval()

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer
    _DEVICE = device

    print(
        f"RemoteCLIP loaded | device={device}"
    )

    return (
        _MODEL,
        _PREPROCESS,
        _TOKENIZER,
        _DEVICE,
    )


# ============================================================
# GEOIMAGE LOADING
# ============================================================

def load_image(
    image_path: str | Path,
) -> Image.Image:
    """
    Load normal images and GeoTIFF satellite imagery.

    GeoTIFF handling supports:
        - 3+ bands -> first 3 bands as RGB
        - 2 bands -> duplicated second channel
        - 1 band -> grayscale duplicated to RGB
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Regular images
    # --------------------------------------------------------

    if image_path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        return Image.open(
            image_path
        ).convert("RGB")

    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    import rasterio

    with rasterio.open(
        image_path
    ) as src:

        if src.count >= 3:

            data = src.read(
                [1, 2, 3]
            )

        elif src.count == 2:

            data = src.read(
                [1, 2]
            )

            data = np.concatenate(
                [
                    data,
                    data[1:2],
                ],
                axis=0,
            )

        else:

            band = src.read(1)

            data = np.stack(
                [
                    band,
                    band,
                    band,
                ],
                axis=0,
            )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Robust percentile normalization
    # --------------------------------------------------------

    rgb = np.zeros_like(
        data,
        dtype=np.float32,
    )

    for i in range(3):

        band = data[i]

        finite = np.isfinite(
            band
        )

        if not finite.any():
            continue

        valid = band[finite]

        low = np.percentile(
            valid,
            2,
        )

        high = np.percentile(
            valid,
            98,
        )

        if high <= low:
            high = low + 1.0

        band = np.clip(
            band,
            low,
            high,
        )

        band = (
            band - low
        ) / (
            high - low
        )

        rgb[i] = band

    rgb = np.clip(
        rgb,
        0.0,
        1.0,
    )

    rgb = (
        rgb * 255.0
    ).astype(
        np.uint8
    )

    rgb = np.transpose(
        rgb,
        (1, 2, 0),
    )

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


# ============================================================
# IMAGE EMBEDDING
# ============================================================

@torch.no_grad()
def encode_image(
    image_path: str | Path,
) -> np.ndarray:
    """
    Generate a normalized RemoteCLIP image embedding.
    """

    (
        model,
        preprocess,
        _,
        device,
    ) = load_remoteclip()

    image = load_image(
        image_path
    )

    tensor = preprocess(
        image
    ).unsqueeze(
        0
    ).to(
        device
    )

    features = model.encode_image(
        tensor
    )

    features = features / features.norm(
        dim=-1,
        keepdim=True,
    )

    embedding = (
        features[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    return embedding


# ============================================================
# TEXT EMBEDDING
# ============================================================

@torch.no_grad()
def encode_text(
    text: str,
) -> np.ndarray:
    """
    Generate a normalized RemoteCLIP text embedding.
    """

    if not text or not text.strip():
        raise ValueError(
            "Text query cannot be empty."
        )

    (
        model,
        _,
        tokenizer,
        device,
    ) = load_remoteclip()

    tokens = tokenizer(
        [text]
    ).to(
        device
    )

    features = model.encode_text(
        tokens
    )

    features = features / features.norm(
        dim=-1,
        keepdim=True,
    )

    embedding = (
        features[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    return embedding


# ============================================================
# BUILD VECTOR INDEX
# ============================================================

def build_image_index(
    image_paths: list[str | Path],
) -> dict[str, Any]:
    """
    Build a lightweight local semantic vector index.

    Instead of FAISS, normalized embeddings are stored in a
    NumPy matrix. Retrieval uses cosine similarity, which is
    equivalent to inner-product search for normalized vectors.
    """

    if not image_paths:
        raise ValueError(
            "No images supplied for indexing."
        )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Building RemoteCLIP index for "
        f"{len(image_paths)} image(s)..."
    )

    embeddings = []
    metadata = []

    for i, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image_path = Path(
            image_path
        ).resolve()

        print(
            f"[{i}/{len(image_paths)}] "
            f"Encoding {image_path}"
        )

        embedding = encode_image(
            image_path
        )

        embeddings.append(
            embedding
        )

        metadata.append(
            {
                "path": str(
                    image_path
                ),
                "name": image_path.name,
            }
        )

    matrix = np.stack(
        embeddings
    ).astype(
        np.float32
    )

    # Extra normalization for numerical stability.
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    matrix = matrix / norms

    np.save(
        EMBEDDINGS_PATH,
        matrix,
    )

    np.save(
        METADATA_PATH,
        np.array(
            metadata,
            dtype=object,
        ),
        allow_pickle=True,
    )

    print(
        f"Embeddings saved: "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Metadata saved: "
        f"{METADATA_PATH}"
    )

    return {
        "status": "success",
        "count": len(
            metadata
        ),
        "dimension": int(
            matrix.shape[1]
        ),
        "embeddings_path": str(
            EMBEDDINGS_PATH
        ),
        "metadata_path": str(
            METADATA_PATH
        ),
    }


# ============================================================
# LOAD VECTOR INDEX
# ============================================================

def load_index() -> tuple[
    np.ndarray,
    list[dict[str, Any]],
]:
    """
    Load the NumPy embedding matrix and metadata.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding index not found: "
            f"{EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: "
            f"{METADATA_PATH}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype(
        np.float32
    )

    metadata = np.load(
        METADATA_PATH,
        allow_pickle=True,
    ).tolist()

    return (
        embeddings,
        metadata,
    )


# ============================================================
# TEXT → IMAGE RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve satellite images matching a natural-language query.

    Pipeline:

        text query
              ↓
        RemoteCLIP text embedding
              ↓
        cosine similarity
              ↓
        ranked satellite scenes
    """

    if not query or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    query_embedding = encode_text(
        query
    )

    embeddings, metadata = load_index()

    # --------------------------------------------------------
    # Cosine similarity
    #
    # Both query and indexed embeddings are normalized.
    # Therefore:
    #
    # cosine_similarity = dot_product
    # --------------------------------------------------------

    scores = (
        embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        -scores
    )

    actual_k = min(
        top_k,
        len(ranked_indices),
    )

    results = []

    for idx in ranked_indices[
        :actual_k
    ]:

        idx = int(
            idx
        )

        item = dict(
            metadata[idx]
        )

        item["score"] = float(
            scores[idx]
        )

        results.append(
            item
        )

    return results


# ============================================================
# IMAGE → IMAGE SIMILARITY
# ============================================================

def compare_images(
    image_a: str | Path,
    image_b: str | Path,
) -> dict[str, Any]:
    """
    Compare two satellite scenes using RemoteCLIP embeddings.
    """

    embedding_a = encode_image(
        image_a
    )

    embedding_b = encode_image(
        image_b
    )

    similarity = float(
        np.dot(
            embedding_a,
            embedding_b,
        )
    )

    return {
        "image_a": str(
            Path(image_a).resolve()
        ),
        "image_b": str(
            Path(image_b).resolve()
        ),
        "cosine_similarity": similarity,
    }


# ============================================================
# DIRECTORY → INDEX → RETRIEVE
# ============================================================

def retrieve_from_directory(
    query: str,
    image_directory: str | Path,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Create an index from a directory and retrieve matching
    satellite images.
    """

    image_directory = Path(
        image_directory
    )

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{image_directory}"
        )

    supported = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    image_paths = sorted(
        [
            path
            for path in image_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in supported
            )
        ]
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in "
            f"{image_directory}"
        )

    build_image_index(
        image_paths
    )

    return retrieve(
        query=query,
        top_k=top_k,
    )


# ============================================================
# STATUS
# ============================================================

def retrieval_status() -> dict[str, Any]:
    """
    Return the current semantic retrieval system status.
    """

    model_ready = (
        _MODEL is not None
    )

    index_ready = (
        EMBEDDINGS_PATH.exists()
        and METADATA_PATH.exists()
    )

    indexed_images = 0

    if index_ready:

        try:

            embeddings, metadata = (
                load_index()
            )

            indexed_images = len(
                metadata
            )

            dimension = int(
                embeddings.shape[1]
            )

        except Exception:

            indexed_images = 0
            dimension = None

    else:

        dimension = None

    return {
        "model": MODEL_NAME,
        "model_loaded": model_ready,
        "device": (
            str(_DEVICE)
            if _DEVICE is not None
            else None
        ),
        "checkpoint": str(
            MODEL_PATH
        ),
        "checkpoint_exists": (
            MODEL_PATH.exists()
        ),
        "index_exists": index_ready,
        "indexed_images": indexed_images,
        "embedding_dimension": dimension,
        "search_backend": "NumPy cosine similarity",
    }