from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = (
    "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
)


@lru_cache(maxsize=1)
def load_vqa_model():
    print("Loading remote-sensing Qwen2-VL...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
    )

    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print("Qwen2-VL loaded.")

    return processor, model, device


def run_vqa(
    image_path: str,
    question: str,
) -> dict:

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Image not found: {image}"
        )

    processor, model, device = (
        load_vqa_model()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image),
                },
                {
                    "type": "text",
                    "text": question,
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

    answer = output_text[0].strip()

    return {
        "answer": answer,
        "confidence": None,
        "model": MODEL_NAME,
        "question": question,
    }