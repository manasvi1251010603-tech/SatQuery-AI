from pathlib import Path

import cv2
import numpy as np


BEFORE = Path("data/demo/adaptformer/before.png")
AFTER = Path("data/demo/adaptformer/after.png")
MASK = Path("data/demo/adaptformer/change_mask.png")

OUTPUT = Path("data/demo/adaptformer/change_overlay.png")


def main() -> None:
    after = cv2.imread(str(AFTER))

    if after is None:
        raise FileNotFoundError(AFTER)

    mask = cv2.imread(
        str(MASK),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(MASK)

    if after.shape[:2] != mask.shape:
        mask = cv2.resize(
            mask,
            (after.shape[1], after.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Create a colored overlay for changed pixels.
    overlay = after.copy()

    changed = mask > 0

    # Use a distinct highlight for the change pixels.
    overlay[changed] = [0, 0, 255]

    # Blend original + highlighted result.
    result = cv2.addWeighted(
        after,
        0.65,
        overlay,
        0.35,
        0,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT),
        result,
    )

    print(f"Saved overlay: {OUTPUT}")

    changed_pixels = int(np.count_nonzero(changed))
    total_pixels = int(mask.size)

    ratio = changed_pixels / total_pixels

    print(
        f"Changed pixels: {changed_pixels:,}"
    )
    print(
        f"Total pixels: {total_pixels:,}"
    )
    print(
        f"Change ratio: {ratio:.4%}"
    )


if __name__ == "__main__":
    main()