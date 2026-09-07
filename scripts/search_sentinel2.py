from datetime import datetime, timezone

import planetary_computer
from pystac_client import Client


CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Pune-area example.
PUNE_LON = 73.8567
PUNE_LAT = 18.5204


def main() -> None:
    catalog = Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={
            "type": "Point",
            "coordinates": [PUNE_LON, PUNE_LAT],
        },
        datetime=(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
        ),
        query={
            "eo:cloud_cover": {
                "lt": 20
            }
        },
        max_items=5,
    )

    items = list(search.items())

    print(f"Found {len(items)} Sentinel-2 scenes.\n")

    for index, item in enumerate(items, start=1):
        print(f"Scene {index}")
        print("ID:", item.id)
        print("Date:", item.datetime)
        print("Cloud cover:", item.properties.get("eo:cloud_cover"))
        print("Available assets:")

        for asset_name in item.assets:
            print("  -", asset_name)

        print()


if __name__ == "__main__":
    main()