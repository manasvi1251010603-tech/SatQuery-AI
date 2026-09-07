from backend.app.models.vqa import run_vqa


def main() -> None:

    result = run_vqa(
        "data/demo/adaptformer/after.png",
        "What objects are visible in this remote-sensing image?",
    )

    print("\n========== SATQUERY VQA ==========")

    print("Question:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nModel:")
    print(result["model"])


if __name__ == "__main__":
    main()