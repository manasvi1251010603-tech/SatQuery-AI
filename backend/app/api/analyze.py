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

from backend.app.models.optical_sar import (
    run_optical_sar,
)

from backend.app.models.retrieval_bridge import (
    run_semantic_retrieval,
)


router = APIRouter(
    prefix="/api",
    tags=["analysis"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    query: str
    images: list[str]


# ============================================================
# GROUNDING TARGET EXTRACTION
# ============================================================

def extract_grounding_target(
    query: str,
) -> str:

    target = (
        query
        .lower()
        .strip()
    )

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

            target = target[
                len(prefix):
            ]

            break

    target = target.strip(
        " ?!.,"
    )

    trailing_phrases = [
        " in this image",
        " in the image",
        " on the image",
        " here",
    ]

    for phrase in trailing_phrases:

        if target.endswith(
            phrase
        ):

            target = target[
                :-len(phrase)
            ].strip()

    if not target:

        target = "object"

    return target


# ============================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================

@router.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> dict[str, Any]:

    # ========================================================
    # 1. BASIC VALIDATION
    # ========================================================

    query = (
        request.query
        .strip()
    )

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # ========================================================
    # 2. PLANNER
    # ========================================================

    try:

        plan = classify_query(
            query=query,
            image_count=len(
                request.images
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Query planning failed: "
                f"{exc}"
            ),
        ) from exc

    task = plan.get(
        "task",
        "UNKNOWN",
    )

    tool = plan.get(
        "tool",
        "unknown",
    )

    # ========================================================
    # 3. IMAGE VALIDATION
    #
    # Semantic Retrieval intentionally accepts zero uploaded
    # images because the query searches the imagery index.
    # ========================================================

    required_images = int(
        plan.get(
            "required_images",
            1,
        )
    )

    if (
        tool != "semantic_retrieval"
        and not request.images
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one image is required "
                "for this workflow."
            ),
        )

    if len(
        request.images
    ) < required_images:

        raise HTTPException(
            status_code=400,
            detail=(
                f"The selected task requires "
                f"{required_images} image(s), "
                f"but {len(request.images)} "
                f"were provided."
            ),
        )

    # ========================================================
    # 4. EXECUTION TRACE
    # ========================================================

    try:

        trace = create_trace(
            plan
        )

    except Exception:

        trace = []

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

    # ========================================================
    # 5. SEMANTIC RETRIEVAL
    # ========================================================

    if tool == "semantic_retrieval":

        try:

            result = run_semantic_retrieval(
                query=query,
                top_k=5,
            )

        except FileNotFoundError as exc:

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

        except ValueError as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Semantic Retrieval "
                    f"validation failed: {exc}"
                ),
            ) from exc

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Semantic Retrieval failed: "
                    f"{exc}"
                ),
            ) from exc

        results = result.get(
            "results",
            [],
        )

        if results:

            top_result = results[0]

            top_name = top_result.get(
                "name",
                "retrieved scene",
            )

            top_score = top_result.get(
                "score",
                0.0,
            )

            answer = (
                f"Found {len(results)} relevant "
                f"satellite scene(s). The top match "
                f"is '{top_name}' with a cosine "
                f"similarity score of "
                f"{float(top_score):.4f}."
            )

        else:

            answer = (
                "No matching satellite scenes "
                "were found in the current imagery index."
            )

        trace.extend(
            [
                {
                    "step": "query_embedding",
                    "status": "completed",
                    "message": (
                        "Converted the natural-language "
                        "query into a RemoteCLIP semantic "
                        "embedding."
                    ),
                },
                {
                    "step": "semantic_search",
                    "status": "completed",
                    "message": (
                        f"Searched the indexed remote-sensing "
                        f"imagery using vector similarity and "
                        f"returned {len(results)} result(s)."
                    ),
                },
                {
                    "step": "ranking",
                    "status": "completed",
                    "message": (
                        "Ranked candidate scenes by cosine "
                        "similarity."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Returned the matching satellite "
                        "scenes and similarity scores as "
                        "retrieval evidence."
                    ),
                },
                {
                    "step": "confidence_estimation",
                    "status": "completed",
                    "message": (
                        "Used the top semantic similarity "
                        "score as a retrieval relevance "
                        "indicator."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "Generated the ranked semantic "
                        "retrieval response."
                    ),
                },
            ]
        )

        return {
            "task": task,
            "tool": tool,
            "query": query,

            "answer": answer,

            "confidence": (
                float(
                    results[0].get(
                        "score",
                        0.0,
                    )
                )
                if results
                else 0.0
            ),

            "confidence_type": (
                "semantic_similarity"
            ),

            "confidence_note": (
                "This score represents semantic "
                "similarity to the indexed scenes; "
                "it is not a calibrated probability."
            ),

            "model": result.get(
                "model",
                "RemoteCLIP ViT-B-32",
            ),

            "search_backend": result.get(
                "search_backend"
            ),

            "results": results,

            "evidence": {
                "retrieved_images": results,
            },

            "trace": trace,
        }

    # ========================================================
    # 6. OPTICAL + SAR
    # ========================================================

    if tool == "optical_sar":

        if len(
            request.images
        ) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Optical-SAR analysis requires "
                    "two images: Optical RGB and SAR."
                ),
            )

        optical_path = (
            request.images[0]
        )

        sar_path = (
            request.images[1]
        )

        try:

            result = run_optical_sar(
                optical_path=optical_path,
                sar_path=sar_path,
                question=query,
                threshold=0.30,
            )

        except FileNotFoundError as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except ValueError as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Optical-SAR validation failed: "
                    f"{exc}"
                ),
            ) from exc

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Optical-SAR analysis failed: "
                    f"{exc}"
                ),
            ) from exc

        trace.extend(
            [
                {
                    "step": "modality_detection",
                    "status": "completed",
                    "message": (
                        "Detected Optical RGB + SAR "
                        "multimodal workflow."
                    ),
                },
                {
                    "step": "task_aware_preprocessing",
                    "status": "completed",
                    "message": (
                        "Applied modality-aware EO/SAR "
                        "preprocessing."
                    ),
                },
                {
                    "step": "model_selection",
                    "status": "completed",
                    "message": (
                        "Selected GalaxEye EfficientNet-B0 "
                        "U-Net EO-SAR specialist model."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "Optical-SAR change detection "
                        "completed."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Generated change mask, "
                        "probability map and evidence overlay."
                    ),
                },
                {
                    "step": "confidence_estimation",
                    "status": "completed",
                    "message": (
                        "Generated model-derived "
                        "confidence indicator."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "Natural-language Optical-SAR "
                        "result generated."
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

            "confidence_type": result.get(
                "confidence_type"
            ),

            "confidence_note": result.get(
                "confidence_note"
            ),

            "change_percentage": result.get(
                "change_percentage"
            ),

            "changed_pixels": result.get(
                "changed_pixels"
            ),

            "valid_pixels": result.get(
                "valid_pixels"
            ),

            "mean_probability": result.get(
                "mean_probability"
            ),

            "changed_region_probability": result.get(
                "changed_region_probability"
            ),

            "model": result.get(
                "model"
            ),

            "device": result.get(
                "device"
            ),

            "threshold": result.get(
                "threshold"
            ),

            "evidence": {
                "change_mask": result.get(
                    "change_mask"
                ),
                "change_probability": result.get(
                    "change_probability"
                ),
                "evidence_image": result.get(
                    "evidence_image"
                ),
            },

            "validation": result.get(
                "validation"
            ),

            "checkpoint_info": result.get(
                "checkpoint_info"
            ),

            "trace": trace,
        }

    # ========================================================
    # 7. MULTITEMPORAL CHANGE DETECTION
    # ========================================================

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
                        "Natural-language change "
                        "result generated."
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

    # ========================================================
    # 8. GROUNDING
    # ========================================================

    if tool == "grounding":

        target = extract_grounding_target(
            query
        )

        try:

            result = run_grounding(
                image_path=request.images[0],
                query=target,
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
                    "step": "preprocessing",
                    "status": "completed",
                    "message": (
                        "Image prepared for "
                        "text-guided region grounding."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "Grounding DINO region detection "
                        "completed."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Detected regions and grounding "
                        "visualization generated."
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

        return {
            "task": task,
            "tool": tool,
            "query": query,

            "target": target,

            "answer": result.get(
                "answer",
                "No grounding result generated.",
            ),

            "confidence": result.get(
                "confidence"
            ),

            "model": result.get(
                "model"
            ),

            "detections": result.get(
                "detections",
                [],
            ),

            "evidence": {
                "grounding_image": result.get(
                    "grounding_image"
                ),
            },

            "trace": trace,
        }

    # ========================================================
    # 9. VQA
    # ========================================================

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
                detail=(
                    "VQA failed: "
                    f"{exc}"
                ),
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

            "evidence": result.get(
                "evidence",
                {},
            ),

            "trace": trace,
        }

    # ========================================================
    # 10. CAPTION / SCENE DESCRIPTION
    # ========================================================

    if (
        task == "CAPTION"
        or tool == "caption"
    ):

        try:

            result = run_vqa(
                image_path=request.images[0],
                question=(
                    "Describe the satellite scene "
                    "concisely, focusing on the main "
                    "land-cover types, structures and "
                    "visible geographic features."
                ),
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
                    "Scene description failed: "
                    f"{exc}"
                ),
            ) from exc

        trace.extend(
            [
                {
                    "step": "preprocessing",
                    "status": "completed",
                    "message": (
                        "Image prepared for "
                        "remote-sensing scene description."
                    ),
                },
                {
                    "step": "inference",
                    "status": "completed",
                    "message": (
                        "Remote-sensing VLM scene "
                        "description completed."
                    ),
                },
                {
                    "step": "evidence_generation",
                    "status": "completed",
                    "message": (
                        "Source image retained as "
                        "visual evidence."
                    ),
                },
                {
                    "step": "response_generation",
                    "status": "completed",
                    "message": (
                        "Scene description generated."
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
                "No description generated.",
            ),

            "confidence": result.get(
                "confidence"
            ),

            "model": result.get(
                "model"
            ),

            "evidence": result.get(
                "evidence",
                {},
            ),

            "trace": trace,
        }

    # ========================================================
    # 11. UNSUPPORTED TOOL
    # ========================================================

    raise HTTPException(
        status_code=400,
        detail=(
            f"SatQuery selected unsupported "
            f"tool '{tool}' for task '{task}'."
        ),
    )