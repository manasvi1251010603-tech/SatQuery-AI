from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


DATA_DIR = Path("data/raw/sentinel2")
OUTPUT = Path("data/demo/sentinel2_false_color.png")


def normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)

    low = np.percentile(band, 2)
    high = np.percentile(band, 98)

    if high <= low:
        return np.zeros_like(band)

    band = np.clip(band, low, high)

    return (band - low) / (high - low)


def main() -> None:
    with rasterio.open(DATA_DIR / "nir.tif") as src:
        nir = src.read(1)

    with rasterio.open(DATA_DIR / "red.tif") as src:
        red = src.read(1)

    with rasterio.open(DATA_DIR / "green.tif") as src:
        green = src.read(1)

    nir = normalize_band(nir)
    red = normalize_band(red)
    green = normalize_band(green)

    false_color = np.dstack((nir, red, green))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(false_color)
    plt.title("Sentinel-2 False-Color Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved false-color image to: {OUTPUT}")


if __name__ == "__main__":
    main()
    