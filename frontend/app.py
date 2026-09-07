from __future__ import annotations

from pathlib import Path

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.72;
        margin-bottom: 28px;
    }

    .result-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .section-title {
        margin-top: 20px;
    }

    .retrieval-card {
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 16px;
    }

    .retrieval-score {
        font-size: 14px;
        opacity: 0.72;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ SatQuery AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Natural-language intelligence for remote-sensing imagery
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Analysis Workflow"
)

analysis_mode = st.sidebar.selectbox(
    "Workflow",
    [
        "Automatic",
        "Semantic Retrieval",
        "Optical + SAR",
        "Multitemporal Change",
        "Grounding / Localization",
        "Remote-Sensing VQA",
    ],
)


# ============================================================
# FILE UPLOAD STATE
# ============================================================

optical_file = None
sar_file = None
uploaded_files = []
use_demo = False


# ============================================================
# FILE UPLOADS
# ============================================================

if analysis_mode == "Semantic Retrieval":

    st.info(
        "Semantic Retrieval searches the indexed "
        "satellite imagery using your natural-language query. "
        "No image upload is required."
    )

    st.success(
        "RemoteCLIP ViT-B-32 semantic search is ready."
    )

elif analysis_mode == "Optical + SAR":

    st.info(
        "Upload a 3-band Optical RGB GeoTIFF and "
        "a 1-band SAR GeoTIFF."
    )

    optical_file = st.file_uploader(
        "Optical RGB image",
        type=["tif", "tiff"],
        key="optical_upload",
    )

    sar_file = st.file_uploader(
        "SAR image",
        type=["tif", "tiff"],
        key="sar_upload",
    )

else:

    # Grounding, VQA, captioning and multitemporal
    # workflows use uploaded imagery.
    uploaded_files = st.file_uploader(
        "Upload satellite image(s)",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
        ],
        accept_multiple_files=True,
        key="generic_upload",
    )


# ============================================================
# OPTICAL + SAR DEMO
# ============================================================

if analysis_mode == "Optical + SAR":

    with st.expander(
        "Use existing Optical + SAR demo"
    ):

        demo_optical = (
            PROJECT_ROOT
            / "data"
            / "demo"
            / "bright"
            / "sample"
            / "marshall-wildfire_00000000_pre_disaster.tif"
        )

        demo_sar = (
            PROJECT_ROOT
            / "data"
            / "demo"
            / "bright"
            / "sample"
            / "marshall-wildfire_00000000_post_disaster.tif"
        )

        if (
            demo_optical.exists()
            and demo_sar.exists()
        ):

            st.success(
                "Real Optical + SAR demo pair found."
            )

            use_demo = st.checkbox(
                "Use this demo pair",
                value=False,
            )

        else:

            st.warning(
                "Optical + SAR demo pair not found."
            )


# ============================================================
# QUERY
# ============================================================

st.subheader(
    "Natural-language query"
)

if analysis_mode == "Semantic Retrieval":

    query_placeholder = (
        "Examples:\n"
        "• Find satellite images of residential areas with buildings.\n"
        "• Search for dense urban scenes.\n"
        "• Find similar satellite imagery showing vegetation."
    )

else:

    query_placeholder = (
        "Examples:\n"
        "• Where are the buildings?\n"
        "• What changed between these two images?\n"
        "• Compare the optical and SAR imagery and identify changed regions.\n"
        "• What is visible in this satellite scene?"
    )

query = st.text_area(
    "Ask SatQuery what you want to analyze",
    placeholder=query_placeholder,
    height=120,
)


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(
    uploaded_file,
) -> str:

    output_path = (
        UPLOAD_DIR
        / uploaded_file.name
    )

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return str(
        output_path
    )


# ============================================================
# AUTOMATIC RETRIEVAL DETECTION
# ============================================================

def looks_like_retrieval_query(
    text: str,
) -> bool:

    lowered = (
        text
        .lower()
        .strip()
    )

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
    ]

    return any(
        term in lowered
        for term in retrieval_terms
    )


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🚀 Analyze",
    type="primary",
    use_container_width=True,
):

    # --------------------------------------------------------
    # Query validation
    # --------------------------------------------------------

    if not query.strip():

        st.error(
            "Please enter a query."
        )

        st.stop()

    image_paths: list[str] = []

    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    if analysis_mode == "Semantic Retrieval":

        # No upload is required.

        image_paths = []


    # ========================================================
    # OPTICAL + SAR
    # ========================================================

    elif analysis_mode == "Optical + SAR":

        if use_demo:

            image_paths = [
                str(demo_optical),
                str(demo_sar),
            ]

        else:

            if not optical_file or not sar_file:

                st.error(
                    "Please upload both the Optical "
                    "RGB image and the SAR image."
                )

                st.stop()

            with st.spinner(
                "Saving Optical and SAR inputs..."
            ):

                optical_path = (
                    save_uploaded_file(
                        optical_file
                    )
                )

                sar_path = (
                    save_uploaded_file(
                        sar_file
                    )
                )

            image_paths = [
                optical_path,
                sar_path,
            ]


    # ========================================================
    # OTHER IMAGE-BASED WORKFLOWS
    # ========================================================

    else:

        if not uploaded_files:

            # ------------------------------------------------
            # Automatic mode can still execute retrieval
            # without forcing the user to upload an image.
            # ------------------------------------------------

            if (
                analysis_mode == "Automatic"
                and looks_like_retrieval_query(
                    query
                )
            ):

                image_paths = []

            else:

                st.error(
                    "Please upload at least one image."
                )

                st.stop()

        else:

            with st.spinner(
                "Saving uploaded imagery..."
            ):

                for uploaded_file in uploaded_files:

                    image_paths.append(
                        save_uploaded_file(
                            uploaded_file
                        )
                    )


    # ========================================================
    # QUERY ENRICHMENT
    # ========================================================

    effective_query = query.strip()


    if analysis_mode == "Semantic Retrieval":

        # Keep the user's retrieval query unchanged.
        effective_query = query.strip()


    elif analysis_mode == "Optical + SAR":

        lowered = effective_query.lower()

        if (
            "optical" not in lowered
            and "sar" not in lowered
        ):

            effective_query = (
                "Perform Optical and SAR analysis. "
                + effective_query
            )


    elif analysis_mode == "Multitemporal Change":

        lowered = effective_query.lower()

        if (
            "change" not in lowered
            and "changed" not in lowered
        ):

            effective_query = (
                "Perform multitemporal change analysis. "
                + effective_query
            )


    elif analysis_mode == "Grounding / Localization":

        lowered = effective_query.lower()

        if (
            "where" not in lowered
            and "locate" not in lowered
            and "highlight" not in lowered
            and "find" not in lowered
            and "identify" not in lowered
        ):

            effective_query = (
                "Locate and highlight "
                + effective_query
            )


    elif analysis_mode == "Remote-Sensing VQA":

        effective_query = (
            effective_query
        )


    # ========================================================
    # API REQUEST
    # ========================================================

    payload = {
        "query": effective_query,
        "images": image_paths,
    }

    try:

        with st.spinner(
            "SatQuery is planning and executing the analysis..."
        ):

            response = requests.post(
                f"{API_URL}/api/analyze",
                json=payload,
                timeout=600,
            )

        if response.status_code != 200:

            st.error(
                f"Analysis failed "
                f"({response.status_code})"
            )

            try:

                detail = (
                    response.json()
                    .get(
                        "detail",
                        response.text,
                    )
                )

            except Exception:

                detail = response.text

            st.code(
                str(detail)
            )

            st.stop()

        result = response.json()

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the SatQuery backend."
        )

        st.code(
            "uvicorn backend.app.main:app --reload"
        )

        st.stop()

    except Exception as exc:

        st.error(
            f"Request failed: {exc}"
        )

        st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "SatQuery Result"
    )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True,
    )

    st.write(
        result.get(
            "answer",
            "No answer generated.",
        )
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Task",
            result.get(
                "task",
                "—",
            ),
        )

    with col2:

        st.metric(
            "Tool",
            result.get(
                "tool",
                "—",
            ),
        )

    with col3:

        confidence = result.get(
            "confidence"
        )

        if confidence is None:

            display = "—"

        else:

            # Retrieval similarity is already a raw
            # similarity score, not a calibrated probability.
            if (
                result.get(
                    "confidence_type"
                )
                == "semantic_similarity"
            ):

                display = (
                    f"{float(confidence):.4f}"
                )

            else:

                display = (
                    f"{float(confidence) * 100:.1f}%"
                )

        st.metric(
            "Confidence / Score",
            display,
        )

    with col4:

        change_percentage = result.get(
            "change_percentage"
        )

        if change_percentage is None:

            display = "—"

        else:

            display = (
                f"{float(change_percentage):.2f}%"
            )

        st.metric(
            "Changed",
            display,
        )


    # ========================================================
    # MODEL
    # ========================================================

    if result.get("model"):

        st.caption(
            f"Specialist model: "
            f"{result.get('model')}"
        )

    if result.get("device"):

        st.caption(
            f"Inference device: "
            f"{result.get('device')}"
        )

    if result.get("search_backend"):

        st.caption(
            f"Search backend: "
            f"{result.get('search_backend')}"
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = result.get(
        "evidence",
        {},
    )


    # ========================================================
    # SEMANTIC RETRIEVAL RESULTS
    # ========================================================

    if (
        result.get("tool")
        == "semantic_retrieval"
    ):

        st.markdown(
            '<h3 class="section-title">'
            '🔎 Semantic Retrieval Results'
            '</h3>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Results are ranked using RemoteCLIP "
            "semantic similarity. The score is not "
            "a calibrated probability."
        )

        retrieval_results = result.get(
            "results",
            evidence.get(
                "retrieved_images",
                [],
            ),
        )

        if retrieval_results:

            for index, item in enumerate(
                retrieval_results,
                start=1,
            ):

                if not isinstance(
                    item,
                    dict,
                ):

                    continue

                image_path = item.get(
                    "path"
                )

                image_name = item.get(
                    "name",
                    (
                        Path(image_path).name
                        if image_path
                        else f"Scene {index}"
                    ),
                )

                score = item.get(
                    "score"
                )

                st.markdown(
                    '<div class="retrieval-card">',
                    unsafe_allow_html=True,
                )

                col_image, col_info = st.columns(
                    [1.3, 2]
                )

                with col_image:

                    if image_path:

                        image_file = Path(
                            image_path
                        )

                        if image_file.exists():

                            st.image(
                                str(image_file),
                                caption=(
                                    f"Rank {index}"
                                ),
                                use_container_width=True,
                            )

                        else:

                            st.warning(
                                "Image file not found."
                            )

                    else:

                        st.info(
                            "No image path returned."
                        )

                with col_info:

                    st.markdown(
                        f"### {index}. {image_name}"
                    )

                    if score is not None:

                        st.metric(
                            "Semantic Similarity",
                            f"{float(score):.4f}",
                        )

                    st.write(
                        "Matched against the natural-language "
                        "query using the RemoteCLIP embedding space."
                    )

                    if image_path:

                        with st.expander(
                            "Source path"
                        ):

                            st.code(
                                str(image_path)
                            )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No matching satellite scenes were "
                "found in the current retrieval index."
            )


    # ========================================================
    # GROUNDING RESULTS
    # ========================================================

    if (
        result.get("tool")
        == "grounding"
    ):

        st.markdown(
            '<h3 class="section-title">'
            '🎯 Grounding Results'
            '</h3>',
            unsafe_allow_html=True,
        )

        grounding_image = evidence.get(
            "grounding_image"
        )

        if grounding_image:

            grounding_path = Path(
                grounding_image
            )

            if grounding_path.exists():

                st.image(
                    str(grounding_path),
                    caption=(
                        "Text-guided region grounding"
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Grounding visualization was "
                    "generated but the image file "
                    "could not be found:"
                )

                st.code(
                    str(grounding_path)
                )

        detections = result.get(
            "detections",
            [],
        )

        if detections:

            st.subheader(
                "Detected Regions"
            )

            for index, detection in enumerate(
                detections,
                start=1,
            ):

                if not isinstance(
                    detection,
                    dict,
                ):

                    continue

                label = detection.get(
                    "label",
                    detection.get(
                        "phrase",
                        "object",
                    ),
                )

                score = detection.get(
                    "score",
                    detection.get(
                        "confidence",
                    ),
                )

                box = detection.get(
                    "box",
                    detection.get(
                        "bbox",
                    ),
                )

                col_a, col_b, col_c = st.columns(3)

                with col_a:

                    st.write(
                        f"**Region {index}**"
                    )

                    st.write(
                        str(label)
                    )

                with col_b:

                    if score is not None:

                        try:

                            st.write(
                                f"Confidence: "
                                f"{float(score) * 100:.1f}%"
                            )

                        except Exception:

                            st.write(
                                f"Confidence: {score}"
                            )

                    else:

                        st.write(
                            "Confidence: —"
                        )

                with col_c:

                    if box is not None:

                        st.write(
                            f"Box: {box}"
                        )

                    else:

                        st.write(
                            "Box: —"
                        )

                st.divider()

        else:

            st.info(
                "No grounding detections were returned."
            )


    # ========================================================
    # OPTICAL + SAR RESULTS
    # ========================================================

    if (
        result.get("tool")
        == "optical_sar"
    ):

        st.markdown(
            '<h3 class="section-title">'
            '🛰️ Optical–SAR Evidence'
            '</h3>',
            unsafe_allow_html=True,
        )

        evidence_image = evidence.get(
            "evidence_image"
        )

        if evidence_image:

            evidence_path = Path(
                evidence_image
            )

            if evidence_path.exists():

                st.image(
                    str(evidence_path),
                    caption=(
                        "Optical + SAR change analysis"
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Optical–SAR evidence image "
                    "was generated but could not "
                    "be found."
                )

        col_a, col_b = st.columns(2)

        with col_a:

            changed_pixels = result.get(
                "changed_pixels"
            )

            if changed_pixels is not None:

                st.metric(
                    "Changed Pixels",
                    f"{int(changed_pixels):,}",
                )

        with col_b:

            valid_pixels = result.get(
                "valid_pixels"
            )

            if valid_pixels is not None:

                st.metric(
                    "Valid Pixels",
                    f"{int(valid_pixels):,}",
                )

        probability_path = evidence.get(
            "change_probability"
        )

        mask_path = evidence.get(
            "change_mask"
        )

        if probability_path:

            probability_file = Path(
                probability_path
            )

            with st.expander(
                "Change Probability GeoTIFF"
            ):

                st.write(
                    str(probability_file)
                )

        if mask_path:

            mask_file = Path(
                mask_path
            )

            with st.expander(
                "Change Mask GeoTIFF"
            ):

                st.write(
                    str(mask_file)
                )


    # ========================================================
    # MULTITEMPORAL RESULTS
    # ========================================================

    if (
        result.get("tool")
        == "change_detection"
    ):

        st.markdown(
            '<h3 class="section-title">'
            '📊 Multitemporal Change Evidence'
            '</h3>',
            unsafe_allow_html=True,
        )

        change_mask = evidence.get(
            "change_mask"
        )

        change_overlay = evidence.get(
            "change_overlay"
        )

        if change_overlay:

            overlay_path = Path(
                change_overlay
            )

            if overlay_path.exists():

                st.image(
                    str(overlay_path),
                    caption="Change overlay",
                    use_container_width=True,
                )

        if change_mask:

            mask_file = Path(
                change_mask
            )

            if mask_file.exists():

                with st.expander(
                    "Change Mask"
                ):

                    st.image(
                        str(mask_file),
                        use_container_width=True,
                    )

        regions = result.get(
            "regions",
            [],
        )

        if regions:

            st.subheader(
                "Changed Regions"
            )

            for index, region in enumerate(
                regions,
                start=1,
            ):

                st.write(
                    f"**Region {index}:** "
                    f"x={region.get('x')} "
                    f"y={region.get('y')} "
                    f"width={region.get('width')} "
                    f"height={region.get('height')}"
                )


    # ========================================================
    # VQA RESULTS
    # ========================================================

    if result.get("tool") == "vqa":

        st.markdown(
            '<h3 class="section-title">'
            '👁️ Remote-Sensing VQA'
            '</h3>',
            unsafe_allow_html=True,
        )

        vqa_evidence = evidence

        source_image = vqa_evidence.get(
            "image",
            vqa_evidence.get(
                "source_image"
            ),
        )

        if source_image:

            source_path = Path(
                source_image
            )

            if source_path.exists():

                st.image(
                    str(source_path),
                    caption="Source satellite imagery",
                    use_container_width=True,
                )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation = result.get(
        "validation"
    )

    if validation:

        with st.expander(
            "Input Validation"
        ):

            st.json(
                validation
            )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_info = result.get(
        "checkpoint_info"
    )

    if checkpoint_info:

        with st.expander(
            "Model Information"
        ):

            st.json(
                checkpoint_info
            )


    # ========================================================
    # CONFIDENCE NOTE
    # ========================================================

    confidence_note = result.get(
        "confidence_note"
    )

    if confidence_note:

        with st.expander(
            "Confidence / Reliability Note"
        ):

            st.write(
                confidence_note
            )


    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    trace = result.get(
        "trace",
        [],
    )

    if trace:

        st.markdown(
            '<h3 class="section-title">'
            '⚙️ Execution Trace'
            '</h3>',
            unsafe_allow_html=True,
        )

        for index, step in enumerate(
            trace,
            start=1,
        ):

            step_name = step.get(
                "step",
                f"step_{index}",
            )

            status = step.get(
                "status",
                "completed",
            )

            message = step.get(
                "message",
                "",
            )

            st.write(
                f"**{index}. {step_name}** "
                f"— `{status}`"
            )

            if message:

                st.caption(
                    message
                )


    # ========================================================
    # RAW RESPONSE
    # ========================================================

    with st.expander(
        "Raw SatQuery Response"
    ):

        st.json(
            result
        )