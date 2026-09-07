from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import yaml


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GALAXEYE_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "galaxeye-eo-sar"
)

GALAXEYE_CONFIG = (
    GALAXEYE_ROOT
    / "config.yaml"
)

GALAXEYE_WEIGHTS = (
    GALAXEYE_ROOT
    / "checkpoints"
    / "efficientnet_b0"
    / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "results"
    / "optical_sar"
)


MODEL_NAME = (
    "GalaxEye EfficientNet-B0 U-Net "
    "EO-SAR Change Detection"
)

DEFAULT_THRESHOLD = 0.30
DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 32


# ============================================================
# IMPORT GALAXEYE DEPENDENCIES
# ============================================================

if str(GALAXEYE_ROOT) not in sys.path:
    sys.path.insert(0, str(GALAXEYE_ROOT))


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    """
    Prefer MPS on Apple Silicon when available.
    Fall back to CPU.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_optical_sar_model():
    """
    Load the exact GalaxEye architecture used during training.

    Architecture:
        SMP U-Net
        EfficientNet-B0 encoder
        5 input channels
        decoder_channels=(128, 64, 32, 16, 8)
        Dropout2d(0.10) before segmentation head

    The checkpoint was verified against this exact architecture.
    """

    import segmentation_models_pytorch as smp

    if not GALAXEYE_CONFIG.exists():
        raise FileNotFoundError(
            f"GalaxEye config not found: {GALAXEYE_CONFIG}"
        )

    if not GALAXEYE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"GalaxEye checkpoint not found: {GALAXEYE_WEIGHTS}"
        )

    with open(GALAXEYE_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=5,
        classes=1,
        decoder_channels=(128, 64, 32, 16, 8),
    )

    # Exact training architecture:
    # model.segmentation_head =
    #     Dropout2d(0.10) + existing segmentation head
    model.segmentation_head = nn.Sequential(
        nn.Dropout2d(0.10),
        model.segmentation_head,
    )

    checkpoint = torch.load(
        GALAXEYE_WEIGHTS,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "GalaxEye checkpoint has unexpected format."
        )

    if "model_state" not in checkpoint:
        raise RuntimeError(
            "GalaxEye checkpoint does not contain "
            "'model_state'."
        )

    # Strict loading is intentional.
    # We verified this architecture against the checkpoint.
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    device = get_device()

    model = model.to(device)
    model.eval()

    checkpoint_info = {
        "epoch": checkpoint.get("epoch"),
        "val_f1": checkpoint.get("val_f1"),
        "val_iou": checkpoint.get("val_iou"),
        "device": str(device),
    }

    print(
        "GalaxEye EO-SAR model loaded | "
        f"device={device} | "
        f"epoch={checkpoint_info['epoch']} | "
        f"val_f1={checkpoint_info['val_f1']} | "
        f"val_iou={checkpoint_info['val_iou']}"
    )

    return model, device, checkpoint_info, cfg


# ============================================================
# GEO-TIFF VALIDATION / LOADING
# ============================================================

def _read_optical(optical_path: Path):
    """
    Read 3-band EO RGB GeoTIFF.
    """

    with rasterio.open(optical_path) as src:
        if src.count != 3:
            raise ValueError(
                "Optical image must contain exactly 3 bands "
                f"(RGB). Found {src.count} bands."
            )

        optical = src.read(
            indexes=(1, 2, 3),
            out_dtype="float32",
        )

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

    return (
        optical,
        profile,
        transform,
        crs,
        width,
        height,
        bounds,
    )


def _read_sar(sar_path: Path):
    """
    Read single-band SAR GeoTIFF.
    """

    with rasterio.open(sar_path) as src:
        if src.count != 1:
            raise ValueError(
                "SAR image must contain exactly 1 band. "
                f"Found {src.count} bands."
            )

        sar = src.read(
            1,
            out_dtype="float32",
        )

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

    return (
        sar,
        transform,
        crs,
        width,
        height,
        bounds,
    )


def validate_pair(
    optical_path: str,
    sar_path: str,
) -> dict[str, Any]:

    optical_file = Path(optical_path)
    sar_file = Path(sar_path)

    if not optical_file.exists():
        raise FileNotFoundError(
            f"Optical file not found: {optical_file}"
        )

    if not sar_file.exists():
        raise FileNotFoundError(
            f"SAR file not found: {sar_file}"
        )

    optical, _, optical_transform, optical_crs, optical_w, optical_h, optical_bounds = (
        _read_optical(optical_file)
    )

    sar, sar_transform, sar_crs, sar_w, sar_h, sar_bounds = (
        _read_sar(sar_file)
    )

    if optical_w != sar_w or optical_h != sar_h:
        raise ValueError(
            "Optical and SAR images must have the same "
            "pixel dimensions. "
            f"Optical={optical_w}x{optical_h}, "
            f"SAR={sar_w}x{sar_h}."
        )

    if optical_crs is not None and sar_crs is not None:
        if optical_crs != sar_crs:
            raise ValueError(
                "Optical and SAR CRS do not match. "
                f"Optical={optical_crs}, SAR={sar_crs}."
            )

    # Transform mismatch is worth warning about but not
    # automatically fatal because some datasets can encode
    # equivalent grids with tiny floating differences.
    transform_match = (
        np.allclose(
            np.asarray(optical_transform),
            np.asarray(sar_transform),
            rtol=1e-6,
            atol=1e-6,
        )
    )

    bounds_match = (
        np.allclose(
            np.asarray(optical_bounds),
            np.asarray(sar_bounds),
            rtol=1e-6,
            atol=1e-3,
        )
    )

    return {
        "optical": {
            "bands": 3,
            "width": optical_w,
            "height": optical_h,
            "crs": str(optical_crs),
            "dtype": str(optical.dtype),
        },
        "sar": {
            "bands": 1,
            "width": sar_w,
            "height": sar_h,
            "crs": str(sar_crs),
            "dtype": str(sar.dtype),
        },
        "same_dimensions": True,
        "same_crs": (
            optical_crs == sar_crs
            if optical_crs is not None and sar_crs is not None
            else None
        ),
        "same_transform": transform_match,
        "same_bounds": bounds_match,
    }


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_eo_sar(
    optical: np.ndarray,
    sar: np.ndarray,
    cfg: dict[str, Any],
):
    """
    Reproduce the GalaxEye training preprocessing.

    Input:
        optical = (3, H, W)
        sar     = (H, W)

    Output:
        5-channel tensor:
            R
            G
            B
            SAR CLAHE
            SAR log
    """

    if optical.ndim != 3 or optical.shape[0] != 3:
        raise ValueError(
            "Optical array must have shape (3, H, W)."
        )

    if sar.ndim != 2:
        raise ValueError(
            "SAR array must have shape (H, W)."
        )

    if optical.shape[1:] != sar.shape:
        raise ValueError(
            "Optical and SAR dimensions do not match."
        )

    clahe_clip_limit = float(
        cfg["data"].get(
            "clahe_clip_limit",
            2.0,
        )
    )

    clahe_tile_size = int(
        cfg["data"].get(
            "clahe_tile_size",
            8,
        )
    )

    # --------------------------------------------------------
    # Validity mask
    # --------------------------------------------------------

    valid_mask = (
        (np.sum(np.abs(optical), axis=0) > 0)
        & (np.abs(sar) > 0)
        & np.isfinite(sar)
    )

    # --------------------------------------------------------
    # Optical normalization
    # --------------------------------------------------------

    optical = np.nan_to_num(
        optical,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    # Original training assumes uint8-like values.
    # Clip to expected range before normalization.
    optical = np.clip(
        optical,
        0.0,
        255.0,
    )

    optical_norm = (
        optical / 255.0
    ).astype(np.float32)

    # --------------------------------------------------------
    # SAR cleaning
    # --------------------------------------------------------

    sar = np.nan_to_num(
        sar,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sar_min = float(np.nanmin(sar))
    sar_max = float(np.nanmax(sar))

    # Convert SAR to 8-bit for CLAHE.
    if sar_max > sar_min:
        sar_8bit = (
            (sar - sar_min)
            / (sar_max - sar_min)
            * 255.0
        ).astype(np.uint8)
    else:
        sar_8bit = np.zeros_like(
            sar,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(
            clahe_tile_size,
            clahe_tile_size,
        ),
    )

    sar_clahe = clahe.apply(
        sar_8bit
    ).astype(np.float32)

    # --------------------------------------------------------
    # Log transform
    # --------------------------------------------------------

    sar_log = np.log1p(
        np.maximum(sar, 0.0)
    )

    log_min = float(
        np.nanmin(sar_log)
    )

    log_max = float(
        np.nanmax(sar_log)
    )

    if log_max > log_min:
        sar_log = (
            (sar_log - log_min)
            / (log_max - log_min)
            * 255.0
        ).astype(np.float32)
    else:
        sar_log = np.zeros_like(
            sar_log,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Normalize SAR channels
    # --------------------------------------------------------

    clahe_norm = (
        sar_clahe / 255.0
    ).astype(np.float32)

    log_norm = (
        sar_log / 255.0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Build 5-channel tensor
    # --------------------------------------------------------

    five_channel = np.concatenate(
        [
            optical_norm,
            clahe_norm[np.newaxis, ...],
            log_norm[np.newaxis, ...],
        ],
        axis=0,
    )

    tensor = torch.from_numpy(
        five_channel
    ).float().unsqueeze(0)

    return tensor, valid_mask, sar_clahe / 255.0


# ============================================================
# TILED INFERENCE
# ============================================================

def tiled_inference(
    model,
    tensor: torch.Tensor,
    device: torch.device,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Run overlapping-tile inference on large satellite images.

    Returns:
        probability map
        binary mask
    """

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be greater than zero."
        )

    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            "overlap must satisfy "
            "0 <= overlap < tile_size."
        )

    _, channels, height, width = tensor.shape

    if channels != 5:
        raise ValueError(
            f"Expected 5 channels, found {channels}."
        )

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    stride = tile_size - overlap

    y_positions = list(
        range(
            0,
            max(1, height - tile_size + 1),
            stride,
        )
    )

    x_positions = list(
        range(
            0,
            max(1, width - tile_size + 1),
            stride,
        )
    )

    if not y_positions:
        y_positions = [0]

    if not x_positions:
        x_positions = [0]

    # Ensure final tiles cover image boundaries.
    if height > tile_size:
        last_y = height - tile_size
        if y_positions[-1] != last_y:
            y_positions.append(last_y)

    if width > tile_size:
        last_x = width - tile_size
        if x_positions[-1] != last_x:
            x_positions.append(last_x)

    model.eval()

    with torch.no_grad():

        for y in y_positions:

            for x in x_positions:

                y1 = y
                x1 = x

                y2 = min(
                    y1 + tile_size,
                    height,
                )

                x2 = min(
                    x1 + tile_size,
                    width,
                )

                # Pad small edge tiles to model size.
                tile_h = y2 - y1
                tile_w = x2 - x1

                tile = tensor[
                    :,
                    :,
                    y1:y2,
                    x1:x2,
                ]

                if (
                    tile_h != tile_size
                    or tile_w != tile_size
                ):
                    padded = torch.zeros(
                        (
                            1,
                            5,
                            tile_size,
                            tile_size,
                        ),
                        dtype=tile.dtype,
                    )

                    padded[
                        :,
                        :,
                        :tile_h,
                        :tile_w,
                    ] = tile

                    tile = padded

                tile = tile.to(device)

                logits = model(tile)

                probabilities = torch.sigmoid(
                    logits
                )[0, 0]

                probabilities = (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )

                probabilities = probabilities[
                    :tile_h,
                    :tile_w,
                ]

                probability_map[
                    y1:y2,
                    x1:x2,
                ] += probabilities

                count_map[
                    y1:y2,
                    x1:x2,
                ] += 1.0

    count_map = np.maximum(
        count_map,
        1.0,
    )

    probability_map = (
        probability_map
        / count_map
    )

    mask = (
        probability_map
        >= DEFAULT_THRESHOLD
    ).astype(np.uint8)

    return probability_map, mask


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_geotiff(
    path: Path,
    data: np.ndarray,
    reference_profile: dict[str, Any],
    dtype: str,
):
    profile = reference_profile.copy()

    profile.update(
        count=1,
        dtype=dtype,
        compress="deflate",
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            data.astype(dtype),
            1,
        )


# ============================================================
# VISUALIZATION
# ============================================================

def save_visual_evidence(
    path: Path,
    optical_rgb: np.ndarray,
    sar_clahe: np.ndarray,
    probability_map: np.ndarray,
    mask: np.ndarray,
    change_percentage: float,
):
    """
    Save one evidence panel for the SatQuery UI.

    Panels:
        1. Optical RGB
        2. SAR
        3. Change probability
        4. Change overlay
    """

    optical_rgb = np.clip(
        optical_rgb,
        0.0,
        1.0,
    )

    overlay = optical_rgb.copy()

    changed = mask > 0

    if np.any(changed):

        overlay[changed, 0] = 1.0
        overlay[changed, 1] *= 0.25
        overlay[changed, 2] *= 0.25

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    axes[0, 0].imshow(
        optical_rgb
    )

    axes[0, 0].set_title(
        "Optical RGB"
    )

    axes[0, 1].imshow(
        sar_clahe,
        cmap="gray",
    )

    axes[0, 1].set_title(
        "SAR"
    )

    axes[1, 0].imshow(
        probability_map,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1, 0].set_title(
        "Change Probability"
    )

    axes[1, 1].imshow(
        overlay
    )

    axes[1, 1].set_title(
        "EO-SAR Change Overlay"
    )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        (
            "SatQuery Optical–SAR Analysis\n"
            f"Detected changed pixels: "
            f"{change_percentage:.2f}%"
        ),
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN API
# ============================================================

def run_optical_sar(
    optical_path: str,
    sar_path: str,
    question: str = "Identify changed regions using optical and SAR imagery.",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:

    optical_file = Path(
        optical_path
    )

    sar_file = Path(
        sar_path
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = validate_pair(
        str(optical_file),
        str(sar_file),
    )

    # --------------------------------------------------------
    # Load imagery
    # --------------------------------------------------------

    (
        optical,
        optical_profile,
        optical_transform,
        optical_crs,
        width,
        height,
        optical_bounds,
    ) = _read_optical(
        optical_file
    )

    (
        sar,
        sar_transform,
        sar_crs,
        sar_width,
        sar_height,
        sar_bounds,
    ) = _read_sar(
        sar_file
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, device, checkpoint_info, cfg = (
        load_optical_sar_model()
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor, valid_mask, sar_clahe = (
        preprocess_eo_sar(
            optical,
            sar,
            cfg,
        )
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------



    probability_map, mask = tiled_inference(
        model=model,
        tensor=tensor,
        device=device,
    )

    # Use caller threshold rather than fixed default.
    mask = (
        probability_map
        >= threshold
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Valid-region filtering
    # --------------------------------------------------------

    mask = (
        mask
        & valid_mask.astype(np.uint8)
    ).astype(np.uint8)

    probability_map = np.where(
        valid_mask,
        probability_map,
        0.0,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_pixels = int(
        np.count_nonzero(valid_mask)
    )

    changed_pixels = int(
        np.count_nonzero(mask)
    )

    if valid_pixels > 0:

        change_percentage = (
            changed_pixels
            / valid_pixels
            * 100.0
        )

        valid_probabilities = (
            probability_map[valid_mask]
        )

        mean_probability = float(
            np.mean(valid_probabilities)
        )

    else:

        change_percentage = 0.0
        mean_probability = 0.0

    changed_probabilities = (
        probability_map[
            mask.astype(bool)
        ]
    )

    if changed_probabilities.size > 0:

        changed_region_probability = float(
            np.mean(
                changed_probabilities
            )
        )

    else:

        changed_region_probability = 0.0

    # This is a heuristic confidence indicator,
    # not a calibrated probability of correctness.
    if changed_pixels > 0:

        model_confidence = (
            changed_region_probability
        )

    else:

        model_confidence = max(
            0.0,
            1.0 - mean_probability,
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = optical_file.stem

    mask_path = (
        OUTPUT_DIR
        / f"{stem}_eo_sar_change_mask.tif"
    )

    probability_path = (
        OUTPUT_DIR
        / f"{stem}_eo_sar_change_probability.tif"
    )

    evidence_path = (
        OUTPUT_DIR
        / f"{stem}_eo_sar_evidence.png"
    )

    # Binary mask GeoTIFF
    mask_profile = optical_profile.copy()

    save_geotiff(
        mask_path,
        mask,
        mask_profile,
        "uint8",
    )

    # Probability GeoTIFF
    save_geotiff(
        probability_path,
        probability_map,
        optical_profile,
        "float32",
    )

    # --------------------------------------------------------
    # RGB visualization
    # --------------------------------------------------------

    optical_rgb = np.clip(
        optical.transpose(1, 2, 0)
        / 255.0,
        0.0,
        1.0,
    )

    save_visual_evidence(
        evidence_path,
        optical_rgb,
        sar_clahe,
        probability_map,
        mask,
        change_percentage,
    )

    # --------------------------------------------------------
    # Natural-language answer
    # --------------------------------------------------------

    query = question.strip()

    lower_query = query.lower()

    if (
        "percentage" in lower_query
        or "percent" in lower_query
        or "how much" in lower_query
        or "extent" in lower_query
    ):

        answer = (
            f"The Optical–SAR model classified "
            f"{change_percentage:.2f}% of the valid "
            f"pixels as changed."
        )

    elif (
        "where" in lower_query
        or "locate" in lower_query
        or "highlight" in lower_query
        or "identify" in lower_query
    ):

        if changed_pixels > 0:

            answer = (
                f"The Optical–SAR model detected "
                f"changed regions covering "
                f"{change_percentage:.2f}% of the "
                f"valid image area. The detected "
                f"regions are highlighted in the "
                f"evidence overlay."
            )

        else:

            answer = (
                "The Optical–SAR model did not "
                "classify any valid pixels as changed."
            )

    else:

        if changed_pixels > 0:

            answer = (
                f"Optical–SAR analysis detected "
                f"potential change across "
                f"{change_percentage:.2f}% of the "
                f"valid pixels."
            )

        else:

            answer = (
                "Optical–SAR analysis did not "
                "detect significant changed regions."
            )

    return {
        "answer": answer,
        "confidence": round(
            float(model_confidence),
            4,
        ),
        "confidence_type": (
            "model-derived heuristic"
        ),
        "confidence_note": (
            "This confidence value is derived from "
            "the model's output probabilities and is "
            "not a calibrated probability of correctness."
        ),
        "change_percentage": round(
            float(change_percentage),
            4,
        ),
        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "mean_probability": round(
            mean_probability,
            4,
        ),
        "changed_region_probability": round(
            changed_region_probability,
            4,
        ),
        "change_mask": str(
            mask_path
        ),
        "change_probability": str(
            probability_path
        ),
        "evidence_image": str(
            evidence_path
        ),
        "model": MODEL_NAME,
        "model_checkpoint": str(
            GALAXEYE_WEIGHTS
        ),
        "device": str(device),
        "threshold": threshold,
        "optical_path": str(
            optical_file
        ),
        "sar_path": str(
            sar_file
        ),
        "validation": validation,
        "checkpoint_info": checkpoint_info,
        "question": query,
    }