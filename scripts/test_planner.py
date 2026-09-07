from backend.app.agent.planner import (
    create_task_plan,
)


def test(
    query: str,
    image_count: int,
) -> None:

    result = create_task_plan(
        query=query,
        image_count=image_count,
    )

    print("\n" + "=" * 60)

    print("QUERY:")
    print(query)

    print("IMAGES:")
    print(image_count)

    print("TASK:")
    print(result["task"])

    print("TOOL:")
    print(result["tool"])

    print("REASON:")
    print(result["reason"])

    print(
        "REQUIRED IMAGES:"
    )
    print(result["required_images"])


def main() -> None:

    test(
        "What changed between these images?",
        2,
    )

    test(
        "Where is the water body?",
        1,
    )

    test(
        "Describe this satellite scene.",
        1,
    )

    test(
        "What is visible in this image?",
        1,
    )

    test(
        "Analyze this region using both optical and SAR imagery.",
        2,
    )


if __name__ == "__main__":
    main()