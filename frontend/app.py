from pathlib import Path

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

UPLOAD_ENDPOINT = f"{API_URL}/api/upload"
ANALYZE_ENDPOINT = f"{API_URL}/api/analyze"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: #f7f9fc;
    }

    /* Main content width */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Title */
    .satquery-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .satquery-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Trace */
    .trace-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }

    .trace-step {
        font-weight: 700;
        color: #111827;
    }

    .trace-message {
        color: #4b5563;
        margin-top: 0.2rem;
    }

    /* Result answer */
    .answer-box {
        background: white;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }

    .answer-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .answer-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.4rem;
    }

    /* Small text */
    .muted {
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="satquery-title">🛰️ SatQuery AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="satquery-subtitle">'
    "Agentic Remote-Sensing Intelligence"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("SatQuery")

    st.markdown(
        """
        **Supported workflows**

        • Single-image VQA  
        • Visual Grounding  
        • Multitemporal Change Analysis

        **Coming next**

        • Optical + SAR  
        • Semantic Retrieval  
        • Advanced evidence fusion
        """
    )

    st.divider()

    st.caption(
        "Prototype • Deadline Dodgers"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_uploaded_paths" not in st.session_state:
    st.session_state.last_uploaded_paths = []


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "1. Upload Satellite Imagery"
    "</div>",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload one or two satellite images",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
    help=(
        "Upload one image for VQA/grounding "
        "or two images for temporal comparison."
    ),
)


# ============================================================
# DISPLAY UPLOADED FILES
# ============================================================

if uploaded_files:

    if len(uploaded_files) > 2:

        st.warning(
            "For this prototype, please upload "
            "a maximum of two images."
        )

        uploaded_files = uploaded_files[:2]

    if len(uploaded_files) == 1:

        st.image(
            uploaded_files[0],
            caption=uploaded_files[0].name,
            width=600,
        )

    elif len(uploaded_files) == 2:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Image 1")

            st.image(
                uploaded_files[0],
                caption=uploaded_files[0].name,
                use_container_width=True,
            )

        with col2:

            st.markdown("### Image 2")

            st.image(
                uploaded_files[1],
                caption=uploaded_files[1].name,
                use_container_width=True,
            )


# ============================================================
# QUESTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "2. Ask SatQuery"
    "</div>",
    unsafe_allow_html=True,
)

query = st.text_area(
    "Natural-language query",
    placeholder=(
        "Examples:\n"
        "• What is visible in this image?\n"
        "• Where are the buildings?\n"
        "• What changed between these two images?\n"
        "• How much of the image changed?"
    ),
    height=110,
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(
    "🔍 Analyze with SatQuery",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_clicked:

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one satellite image."
        )

        st.stop()

    if not query.strip():

        st.error(
            "Please enter a natural-language question."
        )

        st.stop()

    if len(uploaded_files) > 2:

        st.error(
            "This prototype supports a maximum of two images."
        )

        st.stop()

    # --------------------------------------------------------
    # Save uploaded files
    # --------------------------------------------------------

    upload_dir = Path(
        "data/demo/uploads"
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_paths = []

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        suffix = Path(
            uploaded_file.name
        ).suffix.lower()

        # Avoid overwriting same-named files.
        safe_name = (
            f"query_{index + 1}{suffix}"
        )

        output_path = (
            upload_dir / safe_name
        )

        output_path.write_bytes(
            uploaded_file.getbuffer()
        )

        local_paths.append(
            str(output_path)
        )

    st.session_state.last_uploaded_paths = (
        local_paths
    )

    # --------------------------------------------------------
    # Send analysis request
    # --------------------------------------------------------

    payload = {
        "query": query,
        "images": local_paths,
    }

    with st.spinner(
        "SatQuery is analyzing the imagery..."
    ):

        try:

            response = requests.post(
                ANALYZE_ENDPOINT,
                json=payload,
                timeout=300,
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the SatQuery backend.\n\n"
                "Make sure FastAPI is running:\n\n"
                "`uvicorn backend.app.main:app --reload`"
            )

            st.stop()

        except requests.exceptions.Timeout:

            st.error(
                "The analysis timed out. "
                "The AI model may still be processing."
            )

            st.stop()

        except requests.exceptions.RequestException as exc:

            st.error(
                f"Request failed: {exc}"
            )

            st.stop()

    # --------------------------------------------------------
    # Backend errors
    # --------------------------------------------------------

    if response.status_code != 200:

        st.error(
            f"SatQuery backend returned "
            f"HTTP {response.status_code}"
        )

        try:
            error_data = response.json()

            st.code(
                str(error_data)
            )

        except ValueError:

            st.code(
                response.text
            )

        st.stop()

    # --------------------------------------------------------
    # Parse result
    # --------------------------------------------------------

    try:

        result = response.json()

    except ValueError:

        st.error(
            "The backend returned invalid JSON."
        )

        st.code(
            response.text
        )

        st.stop()

    st.session_state.last_result = result


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.last_result

if result:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        "3. SatQuery Analysis Result"
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # BASIC RESULT INFORMATION
    # ========================================================

    task = result.get(
        "task",
        "Unknown",
    )

    tool = result.get(
        "tool",
        "Unknown",
    )

    model = result.get(
        "model"
    )

    answer = result.get(
        "answer",
        "No answer returned.",
    )

    confidence = result.get(
        "confidence"
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-label">SatQuery Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Task",
            task,
        )

    with metric2:

        st.metric(
            "Tool",
            tool,
        )

    with metric3:

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.1%}",
            )

        else:

            st.metric(
                "Confidence",
                "N/A",
            )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if model:

        st.caption(
            f"Specialist model: `{model}`"
        )

    # ========================================================
    # INPUT IMAGERY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        "Input Imagery"
        "</div>",
        unsafe_allow_html=True,
    )

    paths = st.session_state.last_uploaded_paths

    if paths:

        if len(paths) == 1:

            path = Path(paths[0])

            if path.exists():

                st.image(
                    str(path),
                    caption="Input satellite image",
                    use_container_width=True,
                )

        elif len(paths) >= 2:

            col1, col2 = st.columns(2)

            first_path = Path(paths[0])
            second_path = Path(paths[1])

            with col1:

                st.markdown("### Before")

                if first_path.exists():

                    st.image(
                        str(first_path),
                        use_container_width=True,
                    )

            with col2:

                st.markdown("### After")

                if second_path.exists():

                    st.image(
                        str(second_path),
                        use_container_width=True,
                    )

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = result.get(
        "evidence",
        {},
    )

    st.markdown(
        '<div class="section-title">'
        "Evidence"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Change mask
    # --------------------------------------------------------

    change_mask = evidence.get(
        "change_mask"
    )

    if change_mask:

        mask_path = Path(
            change_mask
        )

        if mask_path.exists():

            st.markdown(
                "### AI Change Mask"
            )

            st.image(
                str(mask_path),
                caption=(
                    "Pixels classified as changed "
                    "by the change-detection model."
                ),
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Change overlay
    # --------------------------------------------------------

    change_overlay = evidence.get(
        "change_overlay"
    )

    if change_overlay:

        overlay_path = Path(
            change_overlay
        )

        if overlay_path.exists():

            st.markdown(
                "### Change Evidence Overlay"
            )

            st.image(
                str(overlay_path),
                caption=(
                    "Detected changes highlighted "
                    "on the after image."
                ),
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Grounding result
    # --------------------------------------------------------

    grounding_image = evidence.get(
        "grounding_image"
    )

    if grounding_image:

        grounding_path = Path(
            grounding_image
        )

        if grounding_path.exists():

            st.markdown(
                "### Grounded Regions"
            )

            st.image(
                str(grounding_path),
                caption=(
                    "Regions localized from "
                    "the natural-language query."
                ),
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Generic evidence image
    # --------------------------------------------------------

    generic_image = evidence.get(
        "image"
    )

    if generic_image:

        generic_path = Path(
            generic_image
        )

        if generic_path.exists():

            st.markdown(
                "### Visual Evidence"
            )

            st.image(
                str(generic_path),
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Change percentage
    # --------------------------------------------------------

    change_percentage = result.get(
        "change_percentage"
    )

    if change_percentage is not None:

        st.metric(
            "Detected Changed Pixels",
            f"{float(change_percentage):.2f}%",
        )

    # ========================================================
    # GROUNDING DETECTIONS
    # ========================================================

    detections = result.get(
        "detections",
        [],
    )

    if detections:

        st.markdown(
            '<div class="section-title">'
            "Detected Regions"
            "</div>",
            unsafe_allow_html=True,
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            label = detection.get(
                "label",
                "object",
            )

            score = detection.get(
                "confidence",
                0.0,
            )

            box = detection.get(
                "box"
            )

            col1, col2 = st.columns(
                [3, 2]
            )

            with col1:

                st.write(
                    f"**{index}. {label}**"
                )

            with col2:

                st.write(
                    f"Confidence: {float(score):.2%}"
                )

            if box:

                st.caption(
                    f"Bounding box: {box}"
                )

    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        "SatQuery Execution Trace"
        "</div>",
        unsafe_allow_html=True,
    )

    trace = result.get(
        "trace",
        [],
    )

    if not trace:

        st.info(
            "No execution trace was returned."
        )

    else:

        for step in trace:

            status = step.get(
                "status",
                "unknown",
            )

            step_name = step.get(
                "step",
                "unknown",
            )

            message = step.get(
                "message",
                "",
            )

            if status == "completed":

                icon = "✅"

            elif status == "failed":

                icon = "❌"

            elif status == "pending":

                icon = "⏳"

            else:

                icon = "ℹ️"

            st.markdown(
                f"""
                <div class="trace-box">
                    <div class="trace-step">
                        {icon} {step_name}
                    </div>
                    <div class="trace-message">
                        {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # RAW RESPONSE
    # ========================================================

    with st.expander(
        "Developer: View API Response"
    ):

        st.json(result)

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.info(
        "Upload satellite imagery, enter a question, "
        "and click **Analyze with SatQuery**."
    )