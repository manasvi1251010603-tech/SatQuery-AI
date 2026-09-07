from pathlib import Path
import sys

import numpy as np
import rasterio


def inspect_image(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    with rasterio.open(image_path) as src:
        print("\n========== SATQUERY GEO-TIFF INSPECTOR ==========")

        print(f"File       : {image_path}")
        print(f"Driver     : {src.driver}")
        print(f"Width      : {src.width}")
        print(f"Height     : {src.height}")
        print(f"Bands      : {src.count}")
        print(f"CRS        : {src.crs}")
        print(f"Resolution : {src.res}")
        print(f"Bounds     : {src.bounds}")
        print(f"NoData     : {src.nodata}")
        print(f"DTypes     : {src.dtypes}")

        print("\n--------------- BAND INFORMATION ---------------")

        for band_number in range(1, src.count + 1):
            band = src.read(band_number)

            valid = band[np.isfinite(band)]

            if valid.size == 0:
                print(f"\nBand {band_number}: no valid pixels")
                continue

            print(f"\nBand {band_number}")
            print(f"  Shape   : {band.shape}")
            print(f"  Min     : {valid.min()}")
            print(f"  Max     : {valid.max()}")
            print(f"  Mean    : {valid.mean():.4f}")
            print(f"  Median  : {np.median(valid):.4f}")

        print("\n=================================================\n")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/inspect_geotiff.py <path-to-tif>")
        raise SystemExit(1)

    inspect_image(Path(sys.argv[1]))


if __name__ == "__main__":
    main()