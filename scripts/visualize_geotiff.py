from pathlib import Path

import matplotlib.pyplot as plt
import rasterio


IMAGE_PATH = Path("data/demo/test_satellite.tif")


def main() -> None:
    with rasterio.open(IMAGE_PATH) as src:
        band1 = src.read(1)

    plt.figure(figsize=(8, 8))
    plt.imshow(band1)
    plt.title("SatQuery — Band 1")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()