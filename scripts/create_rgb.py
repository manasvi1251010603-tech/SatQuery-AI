from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


DATA_DIR = Path("data/raw/sentinel2")
OUTPUT = Path("data/demo/sentinel2_rgb.png")


def normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)

    low = np.percentile(band, 2)
    high = np.percentile(band, 98)

    if high <= low:
        return np.zeros_like(band)

    band = np.clip(band, low, high)

    return (band - low) / (high - low)


def main() -> None:
    with rasterio.open(DATA_DIR / "red.tif") as src:
        red = src.read(1)

    with rasterio.open(DATA_DIR / "green.tif") as src:
        green = src.read(1)

    with rasterio.open(DATA_DIR / "blue.tif") as src:
        blue = src.read(1)

    red = normalize_band(red)
    green = normalize_band(green)
    blue = normalize_band(blue)

    rgb = np.dstack((red, green, blue))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("Sentinel-2 RGB Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved RGB image to: {OUTPUT}")


if __name__ == "__main__":
    main()
    