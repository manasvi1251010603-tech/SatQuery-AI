from pathlib import Path

import planetary_computer
import rasterio
from pystac_client import Client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

OUTPUT_DIR = Path("data/raw/sentinel2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENE_ID = "S2B_MSIL2A_20251229T053139_R105_T43QCA_20251229T075410"

BANDS = {
    "B02": "blue.tif",
    "B03": "green.tif",
    "B04": "red.tif",
    "B08": "nir.tif",
}


def main() -> None:
    catalog = Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        ids=[SCENE_ID],
    )

    items = list(search.items())

    if not items:
        raise RuntimeError("Scene not found.")

    item = items[0]

    print("Scene:", item.id)
    print("Date:", item.datetime)

    for band_name, output_name in BANDS.items():
        if band_name not in item.assets:
            raise RuntimeError(f"Asset {band_name} not found.")

        asset = item.assets[band_name]

        output_path = OUTPUT_DIR / output_name

        print(f"\nDownloading {band_name} -> {output_path}")

        with rasterio.open(asset.href) as src:
            profile = src.profile.copy()

            with rasterio.open(output_path, "w", **profile) as dst:
                for window in src.block_windows(1):
                    _, win = window
                    data = src.read(window=win)
                    dst.write(data, window=win)

        print("Saved:", output_path)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()