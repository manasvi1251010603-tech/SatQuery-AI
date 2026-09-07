from backend.app.models.grounding import (
    run_grounding,
)


def main() -> None:

    result = run_grounding(
        "data/demo/adaptformer/after.png",
        "buildings",
    )

    print(
        "\n========== SATQUERY GROUNDING =========="
    )

    print("Question:")
    print(result["query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(
        f"{result['confidence']:.3f}"
    )

    print("\nDetections:")

    for detection in result[
        "detections"
    ]:
        print(
            detection
        )

    print("\nAnnotated image:")
    print(
        result["grounding_image"]
    )


if __name__ == "__main__":
    main()