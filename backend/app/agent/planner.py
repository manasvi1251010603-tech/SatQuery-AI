from __future__ import annotations

from typing import Any


def _contains_any(
    text: str,
    keywords: list[str],
) -> bool:
    return any(
        keyword in text
        for keyword in keywords
    )


def create_task_plan(
    query: str,
    image_count: int,
) -> dict[str, Any]:

    text = (
        query
        .lower()
        .strip()
    )

    # =========================================================
    # 1. OPTICAL + SAR
    # =========================================================

    optical_terms = [
        "optical",
        "eo",
        "electro optical",
        "electro-optical",
        "rgb",
        "sentinel-2",
        "sentinel 2",
    ]

    sar_terms = [
        "sar",
        "synthetic aperture radar",
        "sentinel-1",
        "sentinel 1",
        "radar",
    ]

    multimodal_terms = [
        "compare",
        "comparison",
        "combine",
        "fusion",
        "multimodal",
        "cross modal",
        "cross-modal",
        "both",
        "together",
        "optical and sar",
        "sar and optical",
    ]

    if (
        _contains_any(text, optical_terms)
        and _contains_any(text, sar_terms)
    ) or (
        _contains_any(text, sar_terms)
        and _contains_any(
            text,
            multimodal_terms,
        )
    ):

        return {
            "task": "OPTICAL_SAR",
            "tool": "optical_sar",
            "required_images": 2,
            "reason": (
                "Query requests joint Optical–SAR "
                "remote-sensing analysis."
            ),
            "models": [
                "GalaxEye EfficientNet-B0 U-Net"
            ],
        }

    # =========================================================
    # 2. MULTITEMPORAL CHANGE
    # =========================================================

    temporal_terms = [
        "before",
        "after",
        "pre",
        "post",
        "previous",
        "earlier",
        "later",
        "temporal",
        "time series",
        "multi temporal",
        "multitemporal",
        "change",
        "changed",
        "difference",
        "damage",
        "damaged",
        "compare dates",
        "over time",
    ]

    if (
        image_count >= 2
        and _contains_any(
            text,
            temporal_terms,
        )
    ):

        return {
            "task": "MULTITEMPORAL_CHANGE",
            "tool": "change_detection",
            "required_images": 2,
            "reason": (
                "Multiple temporal observations "
                "were supplied for change analysis."
            ),
            "models": [
                "AdaptFormer LEVIR-CD"
            ],
        }

    # =========================================================
    # 3. TEXT-GUIDED GROUNDING
    # =========================================================

    grounding_terms = [
        "where",
        "locate",
        "location",
        "highlight",
        "identify",
        "bounding box",
        "bounding boxes",
        "region",
        "regions",
        "show me where",
        "point out",
        "localize",
        "localise",
    ]

    # "find" is intentionally NOT treated as a grounding
    # trigger by itself because it is also common in retrieval
    # queries such as "find satellite images of urban areas".
    #
    # A phrase like "find the buildings" is still handled as
    # grounding below through object-oriented wording.

    grounding_object_terms = [
        "find the buildings",
        "find buildings",
        "find roads",
        "find road",
        "find houses",
        "find house",
        "find vehicles",
        "find vehicle",
        "find aircraft",
        "find airplanes",
        "find water bodies",
        "find water body",
        "find forests",
        "find forest",
        "find airports",
        "find airport",
        "find ships",
        "find ship",
    ]

    if (
        _contains_any(
            text,
            grounding_terms,
        )
        or _contains_any(
            text,
            grounding_object_terms,
        )
    ):

        return {
            "task": "GROUNDING",
            "tool": "grounding",
            "required_images": 1,
            "reason": (
                "Query requests localization of an "
                "object or region in the image."
            ),
            "models": [
                "Grounding DINO"
            ],
        }

    # =========================================================
    # 4. SEMANTIC RETRIEVAL
    # =========================================================

    retrieval_terms = [
        "search satellite images",
        "search satellite imagery",
        "search imagery",
        "search scenes",
        "search images",
        "retrieve satellite images",
        "retrieve satellite imagery",
        "retrieve imagery",
        "retrieve scenes",
        "retrieve images",
        "find satellite images",
        "find satellite imagery",
        "find scenes",
        "find similar satellite images",
        "find similar satellite imagery",
        "find similar scenes",
        "find similar imagery",
        "show similar satellite images",
        "show similar satellite imagery",
        "show similar scenes",
        "show similar imagery",
        "look for satellite images",
        "look for satellite imagery",
        "look for scenes",
        "looking for satellite images",
        "looking for satellite imagery",
        "looking for scenes",
        "similar satellite images",
        "similar satellite imagery",
        "similar scenes",
        "similar imagery",
        "semantic search",
        "semantic retrieval",
        "search for",
        "retrieve",
    ]

    if _contains_any(
        text,
        retrieval_terms,
    ):

        return {
            "task": "SEMANTIC_RETRIEVAL",
            "tool": "semantic_retrieval",
            "required_images": 0,
            "reason": (
                "Query requests semantic retrieval of "
                "relevant remote-sensing imagery."
            ),
            "models": [
                "RemoteCLIP ViT-B-32"
            ],
        }

    # =========================================================
    # 5. SCENE CAPTION / DESCRIPTION
    # =========================================================

    caption_terms = [
        "describe",
        "description",
        "scene",
        "caption",
        "summarize",
        "summary",
        "what can you see",
        "what is visible",
    ]

    if _contains_any(
        text,
        caption_terms,
    ):

        return {
            "task": "CAPTION",
            "tool": "caption",
            "required_images": 1,
            "reason": (
                "Query requests a scene-level "
                "description."
            ),
            "models": [
                "Remote-sensing Qwen2-VL 2B"
            ],
        }

    # =========================================================
    # 6. DEFAULT REMOTE-SENSING VQA
    # =========================================================

    return {
        "task": "VQA",
        "tool": "vqa",
        "required_images": 1,
        "reason": (
            "Query is treated as a "
            "remote-sensing visual question."
        ),
        "models": [
            "Remote-sensing Qwen2-VL 2B"
        ],
    }