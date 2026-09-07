from typing import Any


def create_task_plan(
    query: str,
    image_count: int,
) -> dict[str, Any]:
    """
    Convert a natural-language query and the number of
    available images into a structured SatQuery task plan.

    This is the MVP version of the agent planner.
    It uses deterministic rules instead of an LLM.
    """

    q = query.lower().strip()

    # -------------------------------------------------
    # 1. Multitemporal change analysis
    # -------------------------------------------------

    temporal_keywords = [
        "change",
        "changed",
        "changes",
        "difference",
        "differences",
        "compare",
        "comparison",
        "before",
        "after",
        "between",
        "over time",
        "increase",
        "increased",
        "increasing",
        "decrease",
        "decreased",
        "decreasing",
        "construction",
        "growth",
    ]

    if (
        image_count >= 2
        and any(
            keyword in q
            for keyword in temporal_keywords
        )
    ):
        return {
            "task": "MULTITEMPORAL_CHANGE",
            "tool": "change_detection",
            "reason": (
                "Multiple images are available and "
                "the query requests comparison, "
                "change, or temporal analysis."
            ),
            "required_images": 2,
        }

    # -------------------------------------------------
    # 2. Optical + SAR analysis
    # -------------------------------------------------

    optical_sar_keywords = [
        "optical and sar",
        "optical + sar",
        "optical sar",
        "sar and optical",
        "both optical",
        "both sensors",
        "cross modal",
        "cross-modal",
        "multimodal",
        "multi-modal",
    ]

    if any(
        keyword in q
        for keyword in optical_sar_keywords
    ):
        return {
            "task": "OPTICAL_SAR",
            "tool": "optical_sar",
            "reason": (
                "The query requests analysis using "
                "complementary Optical and SAR imagery."
            ),
            "required_images": 2,
        }

    # -------------------------------------------------
    # 3. Region grounding
    # -------------------------------------------------

    grounding_keywords = [
        "where",
        "locate",
        "location",
        "highlight",
        "find",
        "region",
        "identify the location",
        "show me where",
    ]

    if any(
        keyword in q
        for keyword in grounding_keywords
    ):
        return {
            "task": "GROUNDING",
            "tool": "grounding",
            "reason": (
                "The query requests a spatially "
                "localized object or region."
            ),
            "required_images": 1,
        }

    # -------------------------------------------------
    # 4. Captioning
    # -------------------------------------------------

    caption_keywords = [
        "describe",
        "description",
        "scene description",
        "summarize",
        "summary",
        "what does this scene look like",
    ]

    if any(
        keyword in q
        for keyword in caption_keywords
    ):
        return {
            "task": "CAPTION",
            "tool": "caption",
            "reason": (
                "The query requests a description "
                "or summary of the scene."
            ),
            "required_images": 1,
        }

    # -------------------------------------------------
    # 5. Default → VQA
    # -------------------------------------------------

    return {
        "task": "VQA",
        "tool": "vqa",
        "reason": (
            "The query is interpreted as a "
            "single-image visual question."
        ),
        "required_images": 1,
    }