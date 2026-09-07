from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


BEFORE = Path(
    "data/demo/temporal/before_2025-12-09.png"
)

AFTER = Path(
    "data/demo/temporal/after_2025-12-29.png"
)

MASK = Path(
    "data/demo/temporal/adaptformer_change_mask.png"
)

OUTPUT = Path(
    "data/demo/temporal/change_visualization.png"
)


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

    before = cv2.cvtColor(
        before,
        cv2.COLOR_BGR2RGB,
    )

    after = cv2.cvtColor(
        after,
        cv2.COLOR_BGR2RGB,
    )

    # Resize model output to original image size.
    mask = cv2.resize(
        mask,
        (before.shape[1], before.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    changed = mask > 0

    overlay = after.copy()

    # Highlight changed pixels.
    overlay[changed] = [
        255,
        0,
        0,
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )

    axes[0].imshow(before)
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(after)
    axes[1].set_title("After")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Change Mask")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Change Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Saved visualization to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()