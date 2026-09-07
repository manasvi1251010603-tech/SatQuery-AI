from pathlib import Path

import rasterio


IMAGE_PATH = Path("data/demo/test_satellite.tif")


def main() -> None:
    with rasterio.open(IMAGE_PATH) as src:
        print("========== SATQUERY IMAGE INFO ==========")
        print("File:", IMAGE_PATH)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("Data type:", src.dtypes)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bounds:", src.bounds)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)

        for band_number in range(1, src.count + 1):
            band = src.read(band_number)

            print(f"\n--- Band {band_number} ---")
            print("Shape:", band.shape)
            print("Minimum:", band.min())
            print("Maximum:", band.max())
            print("Mean:", band.mean())


if __name__ == "__main__":
    main()
    