from pathlib import Path

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = (
    "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
)

IMAGE_PATH = Path(
    "data/demo/adaptformer/after.png"
)


def main() -> None:

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )

    print("Loading model...")

    model = (
        Qwen2VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
        )
    )

    # Start with CPU for maximum compatibility.
    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print("Model loaded.")
    print("Device:", device)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(IMAGE_PATH),
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this remote-sensing "
                        "image. Identify the main "
                        "objects and land-cover "
                        "features visible."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("\nRunning VQA...")

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n========== SATQUERY VQA ==========")
    print(output_text[0])


if __name__ == "__main__":
    main()