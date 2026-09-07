from pathlib import Path

import cv2
import numpy as np


BEFORE = Path("data/demo/adaptformer/before.png")
AFTER = Path("data/demo/adaptformer/after.png")
MASK = Path("data/demo/adaptformer/change_mask.png")

OUTPUT = Path("data/demo/adaptformer/change_report.png")


def main() -> None:
    before = cv2.imread(str(BEFORE))
    after = cv2.imread(str(AFTER))
    mask = cv2.imread(
        str(MASK),
        cv2.IMREAD_GRAYSCALE,
    )

    if before is None:
        raise FileNotFoundError(BEFORE)

    if after is None:
        raise FileNotFoundError(AFTER)

    if mask is None:
        raise FileNotFoundError(MASK)

    mask = cv2.resize(
        mask,
        (after.shape[1], after.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    changed = mask > 0

    overlay = after.copy()
    overlay[changed] = [0, 0, 255]

    overlay = cv2.addWeighted(
        after,
        0.65,
        overlay,
        0.35,
        0,
    )

    # Make all four images the same size.
    width = min(
        before.shape[1],
        after.shape[1],
        mask.shape[1],
        overlay.shape[1],
    )

    height = min(
        before.shape[0],
        after.shape[0],
        mask.shape[0],
        overlay.shape[0],
    )

    before = cv2.resize(before, (width, height))
    after = cv2.resize(after, (width, height))
    mask = cv2.resize(mask, (width, height))
    overlay = cv2.resize(overlay, (width, height))

    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    report = np.hstack(
        [
            before,
            after,
            mask_bgr,
            overlay,
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUTPUT),
        report,
    )

    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)

    print(
        f"Change percentage: "
        f"{changed_pixels / total_pixels * 100:.2f}%"
    )

    print(
        f"Saved report: {OUTPUT}"
    )


if __name__ == "__main__":
    main()