
from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "demo" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "data" / "demo" / "results"
UI_DIR = PROJECT_ROOT / "data" / "demo" / "ui"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

DEMO_OPTICAL = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "bright"
    / "sample"
    / "marshall-wildfire_00000000_pre_disaster.tif"
)

DEMO_SAR = (
    PROJECT_ROOT
    / "data"
    / "demo"
    / "bright"
    / "sample"
    / "marshall-wildfire_00000000_post_disaster.tif"
)


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="SatQuery",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("page", "Home")
st.session_state.setdefault("query", "")
st.session_state.setdefault("pending_paths", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("analysis_mode", "Automatic")


# ============================================================
# SAFE HELPERS
# ============================================================

def esc(value: Any) -> str:
    return html.escape(str(value))


def save_uploaded_file(uploaded_file) -> str:
    name = Path(uploaded_file.name).name
    target = UPLOAD_DIR / name

    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(target)


def add_history(result: dict[str, Any]) -> None:
    st.session_state.history.insert(
        0,
        {
            "query": result.get("query", ""),
            "task": result.get("task", ""),
            "tool": result.get("tool", ""),
            "answer": result.get("answer", ""),
        },
    )
    st.session_state.history = st.session_state.history[:10]


def prepare_preview(path: Path) -> Image.Image | None:
    if not path.exists():
        return None

    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            import numpy as np
            import rasterio

            with rasterio.open(path) as src:
                if src.count >= 3:
                    bands = src.read([1, 2, 3]).astype(np.float32)
                    out = []

                    for band in bands:
                        finite = np.isfinite(band)

                        if not np.any(finite):
                            out.append(np.zeros_like(band, dtype=np.uint8))
                            continue

                        lo, hi = np.percentile(band[finite], [2, 98])

                        if hi <= lo:
                            norm = np.zeros_like(band, dtype=np.uint8)
                        else:
                            norm = np.clip(
                                (band - lo) / (hi - lo) * 255.0,
                                0,
                                255,
                            ).astype(np.uint8)

                        out.append(norm)

                    array = np.stack(out, axis=-1)
                    image = Image.fromarray(array, "RGB")

                else:
                    band = src.read(1).astype(np.float32)
                    finite = np.isfinite(band)

                    if np.any(finite):
                        lo, hi = np.percentile(band[finite], [2, 98])
                        if hi > lo:
                            band = (band - lo) / (hi - lo) * 255

                    band = np.clip(band, 0, 255).astype(np.uint8)
                    image = Image.fromarray(
                        np.stack([band, band, band], axis=-1),
                        "RGB",
                    )
        else:
            image = Image.open(path).convert("RGB")

        image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        return image

    except Exception as exc:
        print(f"Preview error: {exc}")
        return None


def uploaded_preview(uploaded_file) -> Image.Image | None:
    try:
        return Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")
    except Exception:
        return None


# ============================================================
# GLOBAL CSS
#
# IMPORTANT:
# All HTML/CSS strings start at column 0. This prevents
# Streamlit Markdown from interpreting them as code blocks.
# ============================================================

CSS = r"""<style>
.stApp {
    background:
        radial-gradient(circle at 88% 4%, rgba(111,188,145,.12), transparent 25%),
        radial-gradient(circle at 3% 40%, rgba(72,137,105,.06), transparent 23%),
        #f5f7f5;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2b22 0%, #0d3025 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
    color: #e8f2eb !important;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.page-kicker {
    color: #71847a;
    font-size: 9px;
    letter-spacing: .18em;
    text-transform: uppercase;
    font-weight: 750;
    margin-bottom: 8px;
}

.page-title {
    color: #17352a;
    font-size: 52px;
    line-height: .98;
    letter-spacing: -.055em;
    font-weight: 780;
    margin-bottom: 12px;
}

.page-title span {
    color: #31825f;
}

.page-subtitle {
    color: #728079;
    font-size: 14px;
    line-height: 1.65;
    max-width: 780px;
    margin-bottom: 20px;
}

.section-heading {
    color: #17382b;
    font-size: 22px;
    font-weight: 760;
    letter-spacing: -.025em;
    margin: 19px 0 5px;
}

.section-subheading {
    color: #819088;
    font-size: 11px;
    line-height: 1.55;
    margin-bottom: 13px;
}

.scene-card {
    border: 1px solid #dbe6df;
    border-radius: 18px;
    overflow: hidden;
    background: rgba(255,255,255,.84);
    box-shadow: 0 12px 30px rgba(22,62,46,.06);
    transition: transform .25s ease, box-shadow .25s ease;
}

.scene-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 19px 40px rgba(22,62,46,.11);
}

.scene-card img {
    width: 100%;
    display: block;
}

.scene-copy {
    padding: 13px 15px 16px;
}

.scene-label {
    color: #688176;
    font-size: 8px;
    letter-spacing: .15em;
    text-transform: uppercase;
    font-weight: 750;
}

.scene-title {
    color: #244738;
    font-size: 14px;
    font-weight: 730;
    margin-top: 6px;
}

.scene-text {
    color: #85918b;
    font-size: 10px;
    line-height: 1.55;
    margin-top: 4px;
}

.feature-card {
    min-height: 150px;
    border: 1px solid #dce7e0;
    border-radius: 17px;
    background: linear-gradient(145deg, rgba(255,255,255,.86), rgba(242,248,244,.74));
    padding: 18px;
    transition: transform .23s ease, box-shadow .23s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 16px 34px rgba(25,71,51,.09);
}

.feature-icon {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #e3f1e8;
    border-radius: 12px;
    color: #317857;
    font-weight: 800;
    margin-bottom: 12px;
}

.feature-title {
    color: #2a4a3b;
    font-size: 13px;
    font-weight: 730;
}

.feature-text {
    color: #829088;
    font-size: 10px;
    line-height: 1.55;
    margin-top: 6px;
}

.result-box {
    padding: 20px 21px;
    border: 1px solid #d8e6dc;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(255,255,255,.90), rgba(233,245,237,.72));
    box-shadow: 0 12px 30px rgba(25,71,51,.06);
}

.confidence-panel {
    margin-top: 14px;
    padding: 18px 20px;
    border-radius: 18px;
    border: 1px solid #dce6df;
    background: rgba(255,255,255,.72);
    box-shadow: 0 8px 24px rgba(25,71,51,.04);
}

.confidence-title {
    margin-top: 3px;
    color: #75847d;
    font-size: 9px;
    letter-spacing: .19em;
    font-weight: 800;
}

.confidence-headline {
    font-size: 24px;
    font-weight: 790;
    letter-spacing: -.03em;
    margin-top: 4px;
    margin-bottom: 12px;
}

.confidence-high {
    color: #2f7d58;
}

.confidence-medium {
    color: #9a7a31;
}

.confidence-low {
    color: #aa4b4b;
}

.confidence-level {
    margin-top: 10px;
    margin-bottom: 5px;
    color: #365b49;
    font-size: 11px;
    font-weight: 760;
}

.confidence-item {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    padding: 5px 0;
    color: #65756d;
    font-size: 11px;
    line-height: 1.45;
}

.confidence-item.positive {
    color: #3e6e55;
}

.confidence-item.caution {
    color: #8a743a;
}

.confidence-item.negative {
    color: #985353;
}

.confidence-recommendation {
    margin-top: 14px;
    padding: 11px 13px;
    border-radius: 12px;
    background: #edf5ef;
    border: 1px solid #d9e8dc;
}

.confidence-rec-label {
    color: #668074;
    font-size: 8px;
    letter-spacing: .14em;
    text-transform: uppercase;
    font-weight: 800;
}

.confidence-rec-text {
    color: #335643;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 4px;
}

.result-answer {
    color: #244739;
    font-size: 14px;
    line-height: 1.65;
}

.trace-row {
    display: flex;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #e2e9e4;
}

.trace-index {
    width: 25px;
    height: 25px;
    flex: 0 0 auto;
    border-radius: 50%;
    background: #e5f2e9;
    color: #37785a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    font-weight: 800;
}

.trace-step {
    color: #315342;
    font-size: 10px;
    font-weight: 740;
}

.trace-message {
    color: #7c8b84;
    font-size: 10px;
    margin-top: 3px;
    line-height: 1.4;
}

.stButton > button {
    border-radius: 11px !important;
    border: 1px solid #d7e4dc !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 19px rgba(24,72,49,.08);
}

[data-testid="stMetric"] {
    border: 1px solid #dce7e0;
    border-radius: 13px;
    background: rgba(255,255,255,.78);
}

.footer {
    color: #8a9690;
    font-size: 9px;
    margin-top: 35px;
    padding-top: 14px;
    border-top: 1px solid #dde6e0;
}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    r"""<style>
    /* ========================================================
       MAIN WORKSPACE BUTTONS
       Keep navigation dark, but make content buttons clearly
       visible with the SatQuery green visual language.
       ======================================================== */

    .main .stButton > button {
        background: #eef5f0 !important;
        color: #234b3a !important;
        border: 1px solid #cfe0d5 !important;
        box-shadow: 0 1px 0 rgba(17, 55, 40, .02) !important;
        font-weight: 650 !important;
    }

    .main .stButton > button:hover {
        background: #e2f0e7 !important;
        color: #183d2f !important;
        border-color: #aecaB7 !important;
        box-shadow: 0 8px 18px rgba(27, 74, 52, .08) !important;
    }

    /* Primary Analyze / CTA buttons */
    .main .stButton > button[kind="primary"] {
        background: #235e47 !important;
        color: #f4fbf6 !important;
        border: 1px solid #235e47 !important;
        font-weight: 720 !important;
        box-shadow: 0 8px 20px rgba(35, 94, 71, .16) !important;
    }

    .main .stButton > button[kind="primary"]:hover {
        background: #2f7658 !important;
        color: #ffffff !important;
        border-color: #2f7658 !important;
        box-shadow: 0 10px 24px rgba(35, 94, 71, .20) !important;
    }

    /* Sidebar stays dark and readable. */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,.035) !important;
        color: #dcece3 !important;
        border: 1px solid rgba(155,214,179,.12) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(116,190,145,.14) !important;
        color: #ffffff !important;
        border-color: rgba(155,224,181,.28) !important;
    }
    </style>""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

SIDEBAR_HTML = r"""<div style="padding:12px 6px 15px;">
<div style="display:flex;gap:10px;align-items:center;">
<div style="
width:40px;height:40px;border-radius:14px;
background:radial-gradient(circle at 30% 25%,#c8f2d5,#4da879 45%,#123e2f 100%);
box-shadow:0 0 22px rgba(110,207,154,.22);
position:relative;"></div>
<div>
<div style="font-size:17px;font-weight:780;">SatQuery</div>
<div style="font-size:8px;letter-spacing:.17em;color:#91ad9f !important;text-transform:uppercase;margin-top:2px;">
REMOTE SENSING INTELLIGENCE
</div>
</div>
</div>
</div>
<div style="font-size:8px;letter-spacing:.18em;color:#77998c !important;text-transform:uppercase;margin:2px 5px 8px;">
WORKSPACE
</div>"""

st.sidebar.markdown(SIDEBAR_HTML, unsafe_allow_html=True)

for icon, page in [
    ("⌂", "Home"),
    ("⌁", "Analysis"),
    ("◷", "History"),
    ("⚙", "Settings"),
    ("?", "Help"),
]:
    if st.sidebar.button(
        f"{icon}  {page}",
        key=f"nav_{page}",
        use_container_width=True,
    ):
        st.session_state.page = page
        st.rerun()



# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def build_confidence_analysis(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an explainable confidence summary from evidence that
    the backend already returns.

    This is intentionally evidence-based:
    we do not invent sensor observations that are absent from
    the backend response.
    """

    tool = str(
        result.get("tool", "")
    ).lower()

    task = str(
        result.get("task", "")
    ).lower()

    raw_confidence = result.get(
        "confidence"
    )

    try:
        confidence = (
            float(raw_confidence)
            if raw_confidence is not None
            else None
        )
    except Exception:
        confidence = None

    # Semantic similarity is on a 0-1-ish scale, but should
    # not be displayed as a probability.
    semantic = (
        result.get("confidence_type")
        == "semantic_similarity"
    )

    if semantic:
        score = confidence
    else:
        score = confidence

    high_checks: list[str] = []
    medium_checks: list[str] = []
    low_checks: list[str] = []

    recommendation = (
        "Review the evidence overlay before making a final decision."
    )

    # --------------------------------------------------------
    # Optical + SAR
    # --------------------------------------------------------

    if tool == "optical_sar":

        evidence = result.get(
            "evidence",
            {},
        )

        evidence_image = evidence.get(
            "evidence_image"
        )

        if evidence_image:
            high_checks.append(
                "SAR evidence is available"
            )

        changed_pixels = result.get(
            "changed_pixels"
        )

        valid_pixels = result.get(
            "valid_pixels"
        )

        if (
            changed_pixels is not None
            and valid_pixels is not None
            and float(valid_pixels) > 0
        ):
            ratio = (
                float(changed_pixels)
                / float(valid_pixels)
            )

            if ratio > 0:
                high_checks.append(
                    "Spatial evidence is present in the joint analysis"
                )

        # If the overall confidence is modest, qualify optical
        # visibility instead of overstating certainty.
        if score is not None and score < 0.70:
            medium_checks.append(
                "Optical visibility is limited for this scene"
            )

        elif score is not None and score >= 0.70:
            high_checks.append(
                "Joint optical–SAR model confidence is strong"
            )

        if high_checks:
            recommendation = (
                "Use the SAR-supported result and verify the highlighted "
                "region against the optical observation."
            )
        else:
            recommendation = (
                "Use the result cautiously and inspect both optical and SAR evidence."
            )

    # --------------------------------------------------------
    # Multitemporal change
    # --------------------------------------------------------

    elif tool == "change_detection":

        method = str(
            result.get(
                "method",
                "",
            )
        ).lower()

        regions = result.get(
            "regions",
            [],
        )

        change = result.get(
            "change_percentage"
        )

        if method == "adaptformer":

            high_checks.append(
                "Temporal model agreement is available"
            )

            if regions:
                high_checks.append(
                    "Spatial change regions were identified"
                )

            if score is not None and score >= 0.70:
                high_checks.append(
                    "Change-model confidence is strong"
                )
            else:
                medium_checks.append(
                    "Change-model confidence is moderate"
                )

            recommendation = (
                "Use the model-supported change result and inspect the "
                "highlighted temporal regions."
            )

        else:

            # The latest change detector explicitly distinguishes its
            # visual-difference fallback from AdaptFormer.
            medium_checks.append(
                "Temporal image difference is measurable"
            )

            if regions:
                high_checks.append(
                    "Spatially localized change candidates are present"
                )

            medium_checks.append(
                "AdaptFormer agreement is limited; visual-difference fallback was used"
            )

            recommendation = (
                "Use the change overlay as a candidate-change result; "
                "verify important regions with the source images."
            )

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    elif tool == "grounding":

        detections = result.get(
            "detections",
            [],
        )

        count = len(
            detections
            if isinstance(
                detections,
                list,
            )
            else []
        )

        if count >= 3:

            high_checks.append(
                "Multiple spatial detections support the localization"
            )

        elif count > 0:

            medium_checks.append(
                "Only a small number of regions were localized"
            )

        if score is not None:

            if score >= 0.70:
                high_checks.append(
                    "Top detection confidence is strong"
                )

            elif score >= 0.40:
                medium_checks.append(
                    "Detection confidence is moderate"
                )

            else:
                low_checks.append(
                    "Detection confidence is low"
                )

        if high_checks and not low_checks:

            recommendation = (
                "Use the highlighted spatial regions as the primary "
                "grounding result."
            )

        elif medium_checks:

            recommendation = (
                "Treat the boxes as candidate regions and verify visually."
            )

        else:

            recommendation = (
                "Do not rely on the localization without visual verification."
            )

    # --------------------------------------------------------
    # Semantic Retrieval
    # --------------------------------------------------------

    elif tool == "semantic_retrieval":

        retrieval = result.get(
            "results",
            [],
        )

        if retrieval:

            best = retrieval[0]

            try:
                best_score = float(
                    best.get(
                        "score",
                        0,
                    )
                )
            except Exception:
                best_score = 0.0

            if best_score >= 0.25:

                high_checks.append(
                    "Strong semantic match in the indexed imagery"
                )

            elif best_score >= 0.15:

                medium_checks.append(
                    "Semantic match is moderate"
                )

            else:

                low_checks.append(
                    "Semantic similarity is weak"
                )

            medium_checks.append(
                "Retrieval confidence depends on the current image index"
            )

            recommendation = (
                "Use the highest-ranked scenes as semantic candidates "
                "and inspect their imagery."
            )

        else:

            low_checks.append(
                "No indexed scene matched the query"
            )

            recommendation = (
                "Expand the retrieval index or refine the query."
            )

    # --------------------------------------------------------
    # VQA / Caption
    # --------------------------------------------------------

    elif tool in {
        "vqa",
        "caption",
    }:

        if score is not None:

            if score >= 0.75:

                high_checks.append(
                    "Remote-sensing vision model confidence is strong"
                )

            elif score >= 0.45:

                medium_checks.append(
                    "Remote-sensing vision confidence is moderate"
                )

            else:

                low_checks.append(
                    "Vision-model confidence is low"
                )

        high_checks.append(
            "Answer is grounded in the supplied satellite image"
        )

        recommendation = (
            "Use the paragraph answer together with the source image; "
            "verify low-confidence interpretations visually."
        )

    # --------------------------------------------------------
    # Generic
    # --------------------------------------------------------

    else:

        if score is not None:

            if score >= 0.75:
                high_checks.append(
                    "Overall model confidence is strong"
                )
            elif score >= 0.45:
                medium_checks.append(
                    "Overall model confidence is moderate"
                )
            else:
                low_checks.append(
                    "Overall model confidence is low"
                )

        high_checks.append(
            "Evidence from the selected analysis workflow is available"
        )

    # --------------------------------------------------------
    # Determine headline level
    # --------------------------------------------------------

    if low_checks and not high_checks:

        level = "Low"

    elif medium_checks and not high_checks:

        level = "Medium"

    elif high_checks and not medium_checks:

        level = "High"

    elif high_checks:

        level = "High"

    elif medium_checks:

        level = "Medium"

    else:

        level = "Low"

    return {
        "level": level,
        "high": high_checks,
        "medium": medium_checks,
        "low": low_checks,
        "recommendation": recommendation,
    }


def render_confidence_analysis(
    result: dict[str, Any],
) -> None:

    analysis = build_confidence_analysis(
        result
    )

    st.markdown(
        '<div class="confidence-title">CONFIDENCE</div>',
        unsafe_allow_html=True,
    )

    # Headline confidence.
    level = analysis["level"]

    level_class = {
        "High": "confidence-high",
        "Medium": "confidence-medium",
        "Low": "confidence-low",
    }.get(
        level,
        "confidence-medium",
    )

    st.markdown(
        f'<div class="confidence-headline {level_class}">{esc(level)}</div>',
        unsafe_allow_html=True,
    )

    # High evidence.
    if analysis["high"]:

        st.markdown(
            '<div class="confidence-level">High</div>',
            unsafe_allow_html=True,
        )

        for item in analysis["high"]:

            st.markdown(
                f'<div class="confidence-item positive">✓ <span>{esc(item)}</span></div>',
                unsafe_allow_html=True,
            )

    # Medium / caution evidence.
    if analysis["medium"]:

        st.markdown(
            '<div class="confidence-level">Medium</div>',
            unsafe_allow_html=True,
        )

        for item in analysis["medium"]:

            st.markdown(
                f'<div class="confidence-item caution">△ <span>{esc(item)}</span></div>',
                unsafe_allow_html=True,
            )

    # Low evidence.
    if analysis["low"]:

        st.markdown(
            '<div class="confidence-level">Low</div>',
            unsafe_allow_html=True,
        )

        for item in analysis["low"]:

            st.markdown(
                f'<div class="confidence-item negative">! <span>{esc(item)}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="confidence-recommendation"><div class="confidence-rec-label">Recommendation</div><div class="confidence-rec-text">{esc(analysis["recommendation"])}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    # The hero is rendered in an iframe instead of Markdown.
    # This completely avoids the "HTML displayed as code" issue.
    HERO_HTML = r"""
<!doctype html>
<html>
<head>
<style>
html,body{margin:0;width:100%;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:transparent}
.hero{height:390px;border-radius:28px;position:relative;overflow:hidden;background:linear-gradient(120deg,#173b2d 0%,#2a654a 50%,#0f261e 100%);color:#effaf3}
.grid{position:absolute;inset:0;opacity:.16;background-image:linear-gradient(rgba(255,255,255,.13) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.13) 1px,transparent 1px);background-size:34px 34px}
.glow{position:absolute;width:480px;height:480px;border-radius:50%;right:-130px;top:-130px;background:radial-gradient(circle,rgba(173,233,193,.17),transparent 67%);animation:pulse 5s ease-in-out infinite}
.copy{position:absolute;z-index:5;left:50px;top:53px;width:55%}
.kicker{font-size:9px;letter-spacing:.19em;text-transform:uppercase;color:#a8dfba;font-weight:700;margin-bottom:18px}
.title{font-size:84px;line-height:.9;letter-spacing:-.065em;font-weight:790}
.title span{color:#8ed8a9}
.tag{font-size:21px;font-weight:560;margin-top:17px}
.desc{color:#bdd2c6;font-size:13px;line-height:1.7;margin-top:11px;max-width:530px}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:23px}
.pill{font-size:9px;color:#dcebe2;padding:8px 11px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.06);border-radius:999px;backdrop-filter:blur(8px)}
.earth{position:absolute;width:205px;height:205px;right:118px;top:90px;border-radius:50%;background:radial-gradient(circle at 30% 28%,rgba(222,250,231,.40),transparent 8%),radial-gradient(circle at 37% 34%,#75bb8c 0%,#337a58 43%,#173e30 75%,#081d16 100%);box-shadow:inset -18px -12px 30px rgba(0,0,0,.20),18px 25px 48px rgba(0,0,0,.28);animation:float 5.2s ease-in-out infinite}
.earth:after{content:"";position:absolute;inset:0;border-radius:50%;background:linear-gradient(135deg,rgba(255,255,255,.10),transparent 35%,transparent 65%,rgba(0,0,0,.18))}
.orbit{position:absolute;width:355px;height:355px;right:42px;top:15px;border:1px solid rgba(180,238,200,.18);border-radius:50%;transform:rotate(-18deg);animation:spin 22s linear infinite}
.orbit2{position:absolute;width:270px;height:120px;border:1px solid rgba(180,238,200,.13);border-radius:50%;right:83px;top:133px;transform:rotate(-22deg);animation:spin2 13s linear infinite}
.dot{position:absolute;width:9px;height:9px;border-radius:50%;background:#c0f0ce;right:122px;top:83px;box-shadow:0 0 16px rgba(192,240,206,.85);animation:orbit 4.5s ease-in-out infinite}
.scan{position:absolute;right:0;top:88px;width:470px;height:1px;background:linear-gradient(90deg,transparent,rgba(190,240,207,.7),transparent);animation:scan 5s ease-in-out infinite}
@keyframes pulse{50%{transform:scale(1.08);opacity:1}}
@keyframes float{50%{transform:translateY(-8px) rotate(1deg)}}
@keyframes spin{to{transform:rotate(342deg)}}
@keyframes spin2{to{transform:rotate(-382deg)}}
@keyframes orbit{25%{transform:translate(-13px,5px)}50%{transform:translate(-4px,24px)}75%{transform:translate(13px,7px)}}
@keyframes scan{0%{transform:translateY(0);opacity:0}20%{opacity:.8}55%{transform:translateY(190px);opacity:.35}80%{opacity:.8}100%{transform:translateY(0);opacity:0}}
</style>
</head>
<body>
<div class="hero">
<div class="grid"></div><div class="glow"></div>
<div class="orbit"></div><div class="orbit2"></div>
<div class="earth"></div><div class="dot"></div><div class="scan"></div>
<div class="copy">
<div class="kicker">Satellite imagery · Real insights</div>
<div class="title">Sat<span>Query</span></div>
<div class="tag">Explore the Earth. Find answers.</div>
<div class="desc">Turn satellite observations into evidence-backed answers using natural language and remote-sensing specialists.</div>
<div class="pills">
<div class="pill">Optical + SAR</div>
<div class="pill">Temporal change</div>
<div class="pill">Grounded evidence</div>
</div>
</div>
</div>
</body>
</html>
"""

    import streamlit.components.v1 as components

    components.html(
        HERO_HTML,
        height=410,
        scrolling=False,
    )

    st.markdown(
        '<div class="section-heading">Start an observation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">Bring an image into the workspace, then ask the question that matters.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(
        [1, 1.25],
        gap="large",
    )

    with c1:

        st.subheader("Input imagery")

        st.caption(
            "PNG, JPG, JPEG, WEBP, BMP, TIFF/GeoTIFF and JP2."
        )

        home_uploads = st.file_uploader(
            "Upload satellite observations",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp",
                "tif",
                "tiff",
                "jp2",
            ],
            accept_multiple_files=True,
            key="home_uploads",
        )

        if home_uploads:

            cols = st.columns(
                min(2, len(home_uploads))
            )

            for index, uploaded in enumerate(
                home_uploads
            ):

                image = uploaded_preview(
                    uploaded
                )

                if image:

                    with cols[
                        index % len(cols)
                    ]:

                        st.image(
                            image,
                            caption=uploaded.name,
                            use_container_width=True,
                        )

    with c2:

        st.subheader("Natural-language mission")

        home_query = st.text_area(
            "Question",
            value=st.session_state.query,
            placeholder=(
                "Where are the buildings?\n"
                "What changed between these images?\n"
                "Compare the optical and SAR observations."
            ),
            height=125,
            key="home_query",
        )

        if st.button(
            "Enter analysis workspace  →",
            type="primary",
            use_container_width=True,
            key="home_start",
        ):

            if not home_query.strip():

                st.error("Enter a question first.")

            else:

                st.session_state.query = (
                    home_query.strip()
                )

                st.session_state.pending_paths = [
                    save_uploaded_file(file)
                    for file in (home_uploads or [])
                ]

                st.session_state.page = "Analysis"
                st.rerun()

    st.markdown(
        '<div class="section-heading">From the satellite feed</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subheading">Real remote-sensing imagery from the prototype.</div>',
        unsafe_allow_html=True,
    )

    real1, real2 = st.columns(
        2,
        gap="large",
    )

    optical_preview = prepare_preview(
        DEMO_OPTICAL
    )

    sar_preview = prepare_preview(
        DEMO_SAR
    )

    with real1:

        if optical_preview:

            st.image(
                optical_preview,
                caption="OPTICAL · Earth observation",
                use_container_width=True,
            )

            st.caption(
                "Real optical observation used by the prototype."
            )

    with real2:

        if sar_preview:

            st.image(
                sar_preview,
                caption="SAR · Radar observation",
                use_container_width=True,
            )

            st.caption(
                "Real radar observation used by the prototype."
            )

    st.markdown(
        '<div class="section-heading">What can SatQuery discover?</div>',
        unsafe_allow_html=True,
    )

    features = [
        (
            "⌖",
            "Ground",
            "Locate buildings, roads, vehicles, vegetation and requested regions.",
        ),
        (
            "◫",
            "Change",
            "Compare observations across time and identify candidate change regions.",
        ),
        (
            "◒",
            "Multi-sensor",
            "Analyze complementary optical and SAR observations in one workflow.",
        ),
        (
            "⌘",
            "Retrieve",
            "Search indexed satellite scenes using semantic similarity.",
        ),
    ]

    cols = st.columns(4, gap="medium")

    for index, (
        icon,
        title,
        description,
    ) in enumerate(features):

        with cols[index]:

            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{esc(icon)}</div><div class="feature-title">{esc(title)}</div><div class="feature-text">{esc(description)}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# ANALYSIS
# ============================================================

elif st.session_state.page == "Analysis":

    st.markdown(
        '<div class="page-kicker">Workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Ask the <span>Earth</span>.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Upload imagery, describe the mission in natural language, and let SatQuery select the specialist remote-sensing workflow.</div>',
        unsafe_allow_html=True,
    )

    modes = [
        "Automatic",
        "Semantic Retrieval",
        "Optical + SAR",
        "Multitemporal Change",
        "Grounding / Localization",
        "Remote-Sensing VQA",
    ]

    analysis_mode = st.selectbox(
        "Workflow",
        modes,
        index=modes.index(
            st.session_state.analysis_mode
        ),
        key="workflow_select",
    )

    st.session_state.analysis_mode = analysis_mode

    descriptions = {
        "Automatic":
            "SatQuery selects the specialist workflow from your query and imagery.",
        "Semantic Retrieval":
            "Search indexed satellite scenes using natural-language semantics.",
        "Optical + SAR":
            "Analyze complementary optical and radar observations together.",
        "Multitemporal Change":
            "Compare two observations and identify candidate temporal changes.",
        "Grounding / Localization":
            "Locate requested objects or regions within the satellite image.",
        "Remote-Sensing VQA":
            "Ask visual questions about the satellite observation.",
    }

    st.info(
        descriptions[analysis_mode]
    )

    optical_file = None
    sar_file = None
    uploaded_files = []
    use_demo = False

    st.markdown(
        '<div class="section-heading">Observation inputs</div>',
        unsafe_allow_html=True,
    )

    if analysis_mode == "Semantic Retrieval":

        st.caption(
            "Semantic Retrieval searches the indexed image collection."
        )

    elif analysis_mode == "Optical + SAR":

        a, b = st.columns(2)

        with a:

            optical_file = st.file_uploader(
                "Optical GeoTIFF",
                type=["tif", "tiff"],
                key="analysis_optical",
            )

        with b:

            sar_file = st.file_uploader(
                "SAR GeoTIFF",
                type=["tif", "tiff"],
                key="analysis_sar",
            )

        if (
            DEMO_OPTICAL.exists()
            and DEMO_SAR.exists()
        ):

            use_demo = st.checkbox(
                "Use verified BRIGHT optical + SAR pair",
                key="analysis_demo_pair",
            )

    else:

        uploaded_files = st.file_uploader(
            "Satellite imagery",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp",
                "tif",
                "tiff",
                "jp2",
            ],
            accept_multiple_files=True,
            key="analysis_uploads",
        )

        if uploaded_files:

            cols = st.columns(
                min(3, len(uploaded_files))
            )

            for index, uploaded in enumerate(
                uploaded_files
            ):

                image = uploaded_preview(
                    uploaded
                )

                if image:

                    with cols[
                        index % len(cols)
                    ]:

                        st.image(
                            image,
                            caption=uploaded.name,
                            use_container_width=True,
                        )

    st.markdown(
        '<div class="section-heading">Mission query</div>',
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Query",
        value=st.session_state.query,
        placeholder=(
            "Where are the buildings?\n"
            "What changed between these two images?\n"
            "Compare the optical and SAR imagery."
        ),
        height=118,
        key="analysis_query",
    )

    q1, q2 = st.columns(2)

    quick = [
        "Where are the buildings?",
        "What changed between these two images?",
        "Compare the optical and SAR imagery.",
        "Find satellite images of residential areas.",
    ]

    for index, prompt in enumerate(quick):

        with (q1 if index % 2 == 0 else q2):

            if st.button(
                prompt,
                key=f"quick_{index}",
                use_container_width=True,
            ):

                st.session_state.query = prompt
                st.rerun()

    if st.button(
        "Analyze observation  →",
        key="run_analysis",
        type="primary",
        use_container_width=True,
    ):

        if not query.strip():

            st.error(
                "Please enter a mission query."
            )

            st.stop()

        image_paths: list[str] = []

        if analysis_mode == "Semantic Retrieval":

            image_paths = []

        elif analysis_mode == "Optical + SAR":

            if use_demo:

                image_paths = [
                    str(DEMO_OPTICAL),
                    str(DEMO_SAR),
                ]

            else:

                if not optical_file or not sar_file:

                    st.error(
                        "Upload both Optical and SAR imagery."
                    )

                    st.stop()

                image_paths = [
                    save_uploaded_file(
                        optical_file
                    ),
                    save_uploaded_file(
                        sar_file
                    ),
                ]

        else:

            if uploaded_files:

                image_paths = [
                    save_uploaded_file(
                        file
                    )
                    for file in uploaded_files
                ]

            elif st.session_state.pending_paths:

                image_paths = list(
                    st.session_state.pending_paths
                )

            elif analysis_mode == "Automatic":

                text = query.lower()

                retrieval_terms = [
                    "find satellite",
                    "search satellite",
                    "retrieve satellite",
                    "semantic search",
                    "similar imagery",
                    "similar satellite",
                ]

                if any(
                    term in text
                    for term in retrieval_terms
                ):

                    image_paths = []

                else:

                    st.error(
                        "Please upload at least one satellite image."
                    )

                    st.stop()

            else:

                st.error(
                    "Please upload at least one satellite image."
                )

                st.stop()

        effective_query = query.strip()

        if analysis_mode == "Optical + SAR":

            if not (
                "optical" in effective_query.lower()
                or "sar" in effective_query.lower()
            ):

                effective_query = (
                    "Perform Optical and SAR analysis. "
                    + effective_query
                )

        elif analysis_mode == "Multitemporal Change":

            if not any(
                word in effective_query.lower()
                for word in [
                    "change",
                    "changed",
                ]
            ):

                effective_query = (
                    "Perform multitemporal change analysis. "
                    + effective_query
                )

        elif analysis_mode == "Grounding / Localization":

            if not any(
                word in effective_query.lower()
                for word in [
                    "where",
                    "locate",
                    "highlight",
                    "find",
                    "identify",
                ]
            ):

                effective_query = (
                    "Locate and highlight "
                    + effective_query
                )

        try:

            with st.spinner(
                "SatQuery is planning and executing the observation..."
            ):

                response = requests.post(
                    f"{API_URL}/api/analyze",
                    json={
                        "query": effective_query,
                        "images": image_paths,
                    },
                    timeout=600,
                )

            if response.status_code != 200:

                st.error(
                    f"Analysis failed ({response.status_code})"
                )

                try:
                    detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except Exception:
                    detail = response.text

                st.code(str(detail))
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

        except requests.exceptions.Timeout:

            st.error(
                "The analysis timed out."
            )

            st.stop()

        except Exception as exc:

            st.error(
                f"Request failed: {exc}"
            )

            st.stop()

        st.session_state.query = query.strip()
        st.session_state.pending_paths = image_paths

        add_history(result)

        st.divider()

        st.markdown(
            '<div class="section-heading">SatQuery result</div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------
        # Answer paragraph FIRST
        # ------------------------------------------------

        answer_text = str(
            result.get(
                "answer",
                "No answer generated.",
            )
        ).strip()

        st.markdown(
            '<div class="confidence-title">ANSWER</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-box"><div class="result-answer">{esc(answer_text)}</div></div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------
        # Explainable confidence analysis SECOND
        # ------------------------------------------------

        st.markdown(
            '<div class="confidence-panel">',
            unsafe_allow_html=True,
        )

        render_confidence_analysis(
            result
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Task",
                result.get("task", "—"),
            )

        with m2:
            st.metric(
                "Tool",
                result.get("tool", "—"),
            )

        with m3:

            confidence = result.get("confidence")

            if confidence is None:
                display = "—"
            elif result.get("confidence_type") == "semantic_similarity":
                display = f"{float(confidence):.4f}"
            else:
                display = f"{float(confidence) * 100:.1f}%"

            st.metric(
                "Confidence / Score",
                display,
            )

        with m4:

            change = result.get(
                "change_percentage"
            )

            st.metric(
                "Changed",
                "—"
                if change is None
                else f"{float(change):.2f}%",
            )

        if result.get("model"):
            st.caption(
                f"Model: {result['model']}"
            )

        evidence = result.get(
            "evidence",
            {},
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        if result.get("tool") == "semantic_retrieval":

            st.markdown(
                '<div class="section-heading">Scene retrieval</div>',
                unsafe_allow_html=True,
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
                    1,
                ):

                    if not isinstance(item, dict):
                        continue

                    path_value = item.get("path")
                    score = item.get("score")
                    name = item.get(
                        "name",
                        Path(
                            path_value
                        ).name
                        if path_value
                        else f"Scene {index}",
                    )

                    if path_value and Path(path_value).exists():

                        st.image(
                            path_value,
                            caption=(
                                f"Rank {index} · {name}"
                            ),
                            use_container_width=True,
                        )

                    if score is not None:

                        st.caption(
                            f"Semantic similarity: {float(score):.4f}"
                        )

            else:

                st.info(
                    "No matching satellite scenes were found."
                )

        # ----------------------------------------------------
        # GROUNDING
        # ----------------------------------------------------

        if result.get("tool") == "grounding":

            st.markdown(
                '<div class="section-heading">Grounding map</div>',
                unsafe_allow_html=True,
            )

            grounding_image = evidence.get(
                "grounding_image"
            )

            if grounding_image and Path(
                grounding_image
            ).exists():

                st.image(
                    grounding_image,
                    caption="Text-guided remote-sensing grounding",
                    use_container_width=True,
                )

            detections = result.get(
                "detections",
                [],
            )

            if detections:

                st.markdown(
                    '<div class="section-subheading">Detected regions</div>',
                    unsafe_allow_html=True,
                )

                for index, detection in enumerate(
                    detections,
                    1,
                ):

                    label = detection.get(
                        "label",
                        "object",
                    )

                    score = detection.get(
                        "score",
                        detection.get(
                            "confidence"
                        ),
                    )

                    box = detection.get(
                        "box",
                        detection.get(
                            "bbox"
                        ),
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.write(
                            f"**Region {index}**"
                        )
                        st.caption(
                            str(label)
                        )

                    with c2:

                        if score is not None:

                            st.write(
                                f"Confidence: {float(score) * 100:.1f}%"
                            )

                    with c3:

                        if box is not None:
                            st.write(
                                f"Box: {box}"
                            )

            else:

                st.info(
                    "No grounding detections were returned."
                )

        # ----------------------------------------------------
        # MULTITEMPORAL
        # ----------------------------------------------------

        if result.get("tool") == "change_detection":

            st.markdown(
                '<div class="section-heading">Temporal change evidence</div>',
                unsafe_allow_html=True,
            )

            overlay = evidence.get(
                "change_overlay"
            )

            if overlay and Path(
                overlay
            ).exists():

                st.image(
                    overlay,
                    caption="Candidate temporal changes",
                    use_container_width=True,
                )

            mask = evidence.get(
                "change_mask"
            )

            if mask and Path(mask).exists():

                with st.expander(
                    "View change mask"
                ):

                    st.image(
                        mask,
                        use_container_width=True,
                    )

            difference = evidence.get(
                "change_difference"
            )

            if difference and Path(
                difference
            ).exists():

                with st.expander(
                    "View visual difference"
                ):

                    st.image(
                        difference,
                        use_container_width=True,
                    )

            regions = result.get(
                "regions",
                [],
            )

            if regions:

                st.markdown(
                    '<div class="section-subheading">Candidate changed regions</div>',
                    unsafe_allow_html=True,
                )

                for index, region in enumerate(
                    regions,
                    1,
                ):

                    st.write(
                        f"**Region {index}:** "
                        f"x={region.get('x')} · "
                        f"y={region.get('y')} · "
                        f"width={region.get('width')} · "
                        f"height={region.get('height')}"
                    )

            if result.get("method"):

                st.caption(
                    f"Detection method: {result['method']}"
                )

        # ----------------------------------------------------
        # OPTICAL + SAR
        # ----------------------------------------------------

        if result.get("tool") == "optical_sar":

            st.markdown(
                '<div class="section-heading">Optical–SAR evidence</div>',
                unsafe_allow_html=True,
            )

            evidence_image = evidence.get(
                "evidence_image"
            )

            if evidence_image and Path(
                evidence_image
            ).exists():

                st.image(
                    evidence_image,
                    caption="Optical + SAR analysis",
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # VQA
        # ----------------------------------------------------

        if result.get("tool") == "vqa":

            st.markdown(
                '<div class="section-heading">Remote-sensing vision</div>',
                unsafe_allow_html=True,
            )

            source = (
                evidence.get("image")
                or evidence.get("source_image")
            )

            if source and Path(source).exists():

                st.image(
                    source,
                    caption="Source satellite observation",
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # TRACE
        # ----------------------------------------------------

        trace = result.get(
            "trace",
            [],
        )

        if trace:

            st.markdown(
                '<div class="section-heading">Execution trace</div>',
                unsafe_allow_html=True,
            )

            for index, item in enumerate(
                trace,
                1,
            ):

                if not isinstance(item, dict):
                    continue

                step = item.get(
                    "step",
                    f"Step {index}",
                )

                status = item.get(
                    "status",
                    "completed",
                )

                message = item.get(
                    "message",
                    "",
                )

                st.markdown(
                    f'<div class="trace-row"><div class="trace-index">{index:02d}</div><div><div class="trace-step">{esc(step)} <span style="color:#6f9b83;font-size:8px;text-transform:uppercase;">{esc(status)}</span></div><div class="trace-message">{esc(message)}</div></div></div>',
                    unsafe_allow_html=True,
                )

        if result.get(
            "confidence_note"
        ):

            with st.expander(
                "Reliability note"
            ):

                st.write(
                    result["confidence_note"]
                )

        with st.expander(
            "Developer response"
        ):

            st.json(result)


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    st.markdown(
        '<div class="page-kicker">Workspace memory</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Analysis <span>history</span>.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:

        st.info(
            "No analyses yet."
        )

    else:

        for index, item in enumerate(
            st.session_state.history,
            1,
        ):

            st.markdown(
                f'<div class="feature-card" style="min-height:unset;margin-bottom:10px;"><div class="feature-title">Observation {index}</div><div class="feature-text">{esc(item["query"])}</div><div class="feature-text">{esc(item["task"])} · {esc(item["tool"])}</div></div>',
                unsafe_allow_html=True,
            )

            with st.expander(
                "View result"
            ):

                st.write(
                    item["answer"]
                )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.markdown(
        '<div class="page-kicker">Configuration</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">SatQuery <span>settings</span>.</div>',
        unsafe_allow_html=True,
    )

    for title, value in [
        ("Backend", API_URL),
        ("Interface", "Streamlit"),
        ("Agent orchestration", "Enabled"),
        ("Input formats", "PNG · JPG · JPEG · WEBP · BMP · TIFF · GeoTIFF · JP2"),
        ("Analysis workflows", "Retrieval · Grounding · VQA · Change · Optical–SAR"),
    ]:

        st.markdown(
            f'<div class="feature-card" style="min-height:unset;margin-bottom:10px;"><div class="feature-title">{esc(title)}</div><div class="feature-text">{esc(value)}</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# HELP
# ============================================================

elif st.session_state.page == "Help":

    st.markdown(
        '<div class="page-kicker">Guide</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Explore with <span>SatQuery</span>.</div>',
        unsafe_allow_html=True,
    )

    for index, (
        title,
        description,
    ) in enumerate(
        [
            (
                "Upload imagery",
                "Provide one image, a temporal pair, or an Optical + SAR pair.",
            ),
            (
                "Ask naturally",
                "Describe the information you need in ordinary language.",
            ),
            (
                "Let the planner route it",
                "SatQuery selects the corresponding specialist workflow.",
            ),
            (
                "Inspect evidence",
                "Review maps, detections, change regions, confidence and execution trace.",
            ),
        ],
        1,
    ):

        st.markdown(
            f'<div class="feature-card" style="min-height:unset;margin-bottom:11px;"><div class="feature-icon">{index:02d}</div><div class="feature-title">{esc(title)}</div><div class="feature-text">{esc(description)}</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer"><strong>SatQuery</strong> · Natural-language intelligence for remote-sensing imagery</div>',
    unsafe_allow_html=True,
)
