from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


IMAGE_PATH = Path("data/demo/test_satellite.tif")


def normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)

    min_value = band.min()
    max_value = band.max()

    if max_value == min_value:
        return np.zeros_like(band)

    return (band - min_value) / (max_value - min_value)


def main() -> None:
    with rasterio.open(IMAGE_PATH) as src:
        red = normalize_band(src.read(3))
        green = normalize_band(src.read(2))
        blue = normalize_band(src.read(1))

    rgb = np.dstack((red, green, blue))

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("SatQuery — RGB Composite")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
    