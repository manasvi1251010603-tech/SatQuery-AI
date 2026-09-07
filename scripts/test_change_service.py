from backend.app.models.change_detection import (
    run_change_detection,
)


def main() -> None:

    result = run_change_detection(
        "data/demo/adaptformer/before.png",
        "data/demo/adaptformer/after.png",
        "What changed between these images?",
    )

    print("\n========== SATQUERY CHANGE RESULT ==========")

    print("Answer:")
    print(result["answer"])

    print(
        "\nChange percentage:",
        f"{result['change_percentage']:.2f}%",
    )

    print(
        "\nModel:",
        result["model"],
    )

    print(
        "\nMask:",
        result["change_mask"],
    )

    print(
        "\nOverlay:",
        result["change_overlay"],
    )

    print(
        "\nRegions detected:",
        len(result["regions"]),
    )


if __name__ == "__main__":
    main()