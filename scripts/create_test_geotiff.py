from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


OUTPUT = Path("data/demo/test_satellite.tif")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    height = 512
    width = 512
    bands = 4

    rng = np.random.default_rng(42)

    # Create four synthetic bands.
    data = rng.integers(
        low=0,
        high=10000,
        size=(bands, height, width),
        dtype=np.uint16,
    )

    # Example geospatial placement.
    transform = from_origin(
        73.8567,   # west
        18.5204,   # north
        0.0001,    # pixel width
        0.0001,    # pixel height
    )

    with rasterio.open(
        OUTPUT,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for band_index in range(bands):
            dst.write(data[band_index], band_index + 1)

    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()