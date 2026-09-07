from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.agent.router import classify_query
from backend.app.agent.trace import create_trace
from backend.app.models.change_detection import (
    run_change_detection,
)
from backend.app.models.grounding import (
    run_grounding,
)
from backend.app.models.vqa import (
    run_vqa,
)


router = APIRouter(
    prefix="/api",
    tags=["analysis"],
)


class AnalyzeRequest(BaseModel):
    query: str
    images: list[str]


def extract_grounding_target(query: str) -> str:
    """
    Convert a natural-language grounding request into
    a short target phrase for Grounding DINO.

    Examples:
        "Where are the buildings?"
            -> "buildings"

        "Highlight the water body."
            -> "water body"

        "Find the roads."
            -> "roads"
    """

    target = query.lower().strip()

    prefixes = [
        "where are the ",
        "where is the ",
        "where are ",
        "where is ",
        "locate the ",
        "locate ",
        "highlight the ",
        "highlight ",
        "find the ",
        "find ",
        "show me the ",
        "show me ",
        "identify the ",
        "identify ",
    ]

    for prefix in prefixes:
        if target.startswith(prefix):
            target = target[len(prefix):]
            break

    target = target.strip(" ?.!,")

    # Remove common trailing phrases.
    trailing_phrases = [
        " in this image",
        " in the image",
        " on the image",
        " here",
    ]

    for phrase in trailing_phrases:
        if target.endswith(phrase):
            target = target[: -len(phrase)].strip()

    if not target:
        target = "object"

    return target


@router.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> dict[str, Any]:

    # =========================================================
    # 1. BASIC VALIDATION
    # =========================================================

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if not request.images:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required.",
        )

    # =========================================================
    # 2. SATQUERY PLANNER
    # =========================================================

    plan = classify_query(
        query=query,
        image_count=len(request.images),
    )

    task = plan["task"]
    tool = plan["tool"]

    # =========================================================
    # 3. EXECUTION TRACE
    # =========================================================

    trace = create_trace(plan)

    # =========================================================
    # 4. INPUT COUNT VALIDATION
    # =========================================================

    required_images = plan.get(
        "required_images",
        1,
    )

    if len(request.images) < required_images:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The selected task requires "
                f"{required_images} image(s), but "
                f"{len(request.images)} were provided."
            ),
        )

    trace.append(
        {
            "step": "input_validation",
            "status": "completed",
            "message": (
                f"{len(request.images)} image(s) "
                "available for the selected workflow."
            ),
        }
    )

    # =========================================================
    # 5. MULTITEMPORAL CHANGE DETECTION
    # =========================================================

    if tool == "change_detection":

        try:
            result = run_change_detection(
                before_path=request.images[0],
                after_path=request.images[1],
                question=query,
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Change detection failed: "
                    f"{exc}"
                ),
            ) from exc

        trace.extend(
            [
                {
                    "step": "preprocessing",
                    "status": "completed",
                    "message": (
                        "Temporal imagery prepared "
                        "for change detection."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "AdaptFormer change detection "
                        "completed."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Change mask, overlay and "
                        "changed regions generated."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "Natural-language result "
                        "generated."
                    ),
                },
            ]
        )

        return {
            "task": task,
            "tool": tool,
            "query": query,
            "answer": result.get(
                "answer",
                "No answer generated.",
            ),
            "confidence": result.get(
                "confidence"
            ),
            "change_percentage": result.get(
                "change_percentage"
            ),
            "regions": result.get(
                "regions",
                [],
            ),
            "model": result.get(
                "model"
            ),
            "evidence": {
                "change_mask": result.get(
                    "change_mask"
                ),
                "change_overlay": result.get(
                    "change_overlay"
                ),
            },
            "trace": trace,
        }

    # =========================================================
    # 6. VQA
    # =========================================================

    if tool == "vqa":

        try:
            result = run_vqa(
                image_path=request.images[0],
                question=query,
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"VQA failed: {exc}",
            ) from exc

        trace.extend(
            [
                {
                    "step": "preprocessing",
                    "status": "completed",
                    "message": (
                        "Image prepared for "
                        "remote-sensing VQA."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "Remote-sensing Qwen2-VL "
                        "completed VQA."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Input image retained as "
                        "visual evidence."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "VQA response generated."
                    ),
                },
            ]
        )

        return {
            "task": task,
            "tool": tool,
            "query": query,
            "answer": result.get(
                "answer",
                "No answer generated.",
            ),
            "confidence": result.get(
                "confidence"
            ),
            "model": result.get(
                "model"
            ),
            "evidence": {
                "image": request.images[0],
            },
            "trace": trace,
        }

    # =========================================================
    # 7. GROUNDING
    # =========================================================

    if tool == "grounding":

        # Convert:
        # "Where are the buildings?"
        # into:
        # "buildings"
        grounding_target = extract_grounding_target(
            query
        )

        try:
            result = run_grounding(
                image_path=request.images[0],
                query=grounding_target,
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Grounding failed: "
                    f"{exc}"
                ),
            ) from exc

        trace.extend(
            [
                {
                    "step": "query_refinement",
                    "status": "completed",
                    "message": (
                        f"Grounding target extracted: "
                        f"'{grounding_target}'."
                    ),
                },
                {
                    "step": "preprocessing",
                    "status": "completed",
                    "message": (
                        "Image prepared for "
                        "text-guided grounding."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "Grounding DINO completed "
                        "text-guided localization."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Grounded regions and "
                        "annotated evidence generated."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "Grounding result generated."
                    ),
                },
            ]
        )

        detections = result.get(
            "detections",
            [],
        )

        # More useful answer depending on detection result.
        if detections:
            answer = (
                f"Detected {len(detections)} "
                f"candidate region(s) for "
                f"'{grounding_target}'."
            )
        else:
            answer = (
                f"No confident '{grounding_target}' "
                "regions were detected."
            )

        return {
            "task": task,
            "tool": tool,
            "query": query,
            "grounding_target": grounding_target,
            "answer": answer,
            "confidence": result.get(
                "confidence"
            ),
            "model": result.get(
                "model"
            ),
            "detections": detections,
            "evidence": {
                "grounding_image": result.get(
                    "grounding_image"
                ),
            },
            "trace": trace,
        }

    # =========================================================
    # 8. CAPTIONING
    # =========================================================

    if tool == "caption":

        trace.append(
            {
                "step": "inference",
                "status": "pending",
                "message": (
                    "Remote-sensing captioning "
                    "model is not connected yet."
                ),
            }
        )

        return {
            "task": task,
            "tool": tool,
            "query": query,
            "answer": (
                "Remote-sensing captioning "
                "is not connected yet."
            ),
            "confidence": None,
            "model": None,
            "evidence": {},
            "trace": trace,
        }

    # =========================================================
    # 9. OPTICAL + SAR
    # =========================================================

    if tool == "optical_sar":

        trace.append(
            {
                "step": "inference",
                "status": "pending",
                "message": (
                    "Optical-SAR specialist "
                    "model is not connected yet."
                ),
            }
        )

        return {
            "task": task,
            "tool": tool,
            "query": query,
            "answer": (
                "Optical-SAR analysis "
                "is not connected yet."
            ),
            "confidence": None,
            "model": None,
            "evidence": {},
            "trace": trace,
        }

    # =========================================================
    # 10. UNKNOWN TOOL
    # =========================================================

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported analysis tool: {tool}"
        ),
    )