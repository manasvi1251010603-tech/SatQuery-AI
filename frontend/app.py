from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "demo" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
# PAGE
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

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "query" not in st.session_state:
    st.session_state.query = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_paths" not in st.session_state:
    st.session_state.pending_paths = []


# ============================================================
# HELPERS
# ============================================================

def esc(value: Any) -> str:
    return html.escape(str(value))


def save_uploaded_file(uploaded_file) -> str:
    output_path = UPLOAD_DIR / uploaded_file.name
    with open(output_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(output_path)


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


def looks_like_retrieval_query(text: str) -> bool:
    lowered = text.lower().strip()
    terms = [
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
        "looking for satellite images",
        "looking for satellite imagery",
        "similar satellite images",
        "similar satellite imagery",
        "similar scenes",
        "similar imagery",
        "semantic search",
        "semantic retrieval",
    ]
    return any(term in lowered for term in terms)


def render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        return

    st.markdown(
        '<div class="section-heading">Execution Trace</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Observable steps performed by the SatQuery orchestration layer."
    )

    pieces = ['<div class="trace-card">']
    for index, step in enumerate(trace, start=1):
        name = esc(step.get("step", f"step_{index}"))
        status = esc(step.get("status", "completed"))
        message = esc(step.get("message", ""))
        pieces.append(
            f'<div class="trace-row">'
            f'<span class="trace-dot"></span>'
            f'<div><div class="trace-title">{index}. {name}'
            f'<span class="trace-status">{status}</span></div>'
            f'<div class="trace-message">{message}</div></div>'
            f'</div>'
        )
    pieces.append("</div>")
    st.markdown("".join(pieces), unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
.stApp {
    background: radial-gradient(circle at 68% 4%, rgba(67,134,98,.09), transparent 24%),
                radial-gradient(circle at 4% 85%, rgba(27,91,68,.07), transparent 25%),
                #f5f7f4;
    color: #17261f;
}
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1450px; padding-top: .9rem; padding-bottom: 3rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172a24 0%, #10221d 100%);
    border-right: 1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] > div { padding: 16px 13px; }
.sidebar-brand { display:flex; align-items:center; gap:10px; padding:7px 4px 18px; }
.logo-mark {
    position:relative; width:38px; height:38px; border-radius:12px;
    background: radial-gradient(circle at 32% 28%, #c6f5d6 0, #75d39a 18%, #27775a 56%, #15392e 100%);
    box-shadow: 0 8px 22px rgba(28,101,72,.28);
}
.logo-mark:before { content:""; position:absolute; width:43px; height:16px; left:-3px; top:11px; border:1px solid rgba(231,255,240,.62); border-radius:50%; transform:rotate(-20deg); }
.logo-mark:after { content:""; position:absolute; width:6px; height:6px; left:16px; top:16px; border-radius:50%; background:white; box-shadow:0 0 10px rgba(221,255,234,.8); }
.brand-name { font-family:"Space Grotesk",sans-serif; font-weight:700; font-size:20px; color:#f6fbf8; letter-spacing:-.04em; }
.brand-subtitle { margin-top:3px; font-size:8px; text-transform:uppercase; letter-spacing:.14em; color:rgba(224,244,234,.43); }
.nav-caption { margin:5px 4px 7px; font-size:9px; text-transform:uppercase; letter-spacing:.14em; color:rgba(228,245,236,.38); }
.nav-wrap button { border:0 !important; background:transparent !important; color:rgba(238,248,242,.69) !important; justify-content:flex-start !important; box-shadow:none !important; }
.nav-active button { background:rgba(123,192,154,.16) !important; color:#f2fff6 !important; }
.sidebar-system { margin-top:16px; padding:12px; border-radius:13px; border:1px solid rgba(220,245,231,.08); background:rgba(255,255,255,.025); }
.sidebar-system small { color:rgba(225,242,234,.42); font-size:9px; text-transform:uppercase; letter-spacing:.13em; }
.sidebar-online { margin-top:7px; color:#92e6ae; font-size:11px; font-weight:600; }
.sidebar-online i { display:inline-block; width:6px; height:6px; margin-right:6px; border-radius:50%; background:#70e39c; box-shadow:0 0 9px rgba(112,227,156,.75); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:.45} 50%{opacity:1} }

/* Hero */
.hero {
    position:relative; overflow:hidden; min-height:380px; border-radius:28px;
    border:1px solid rgba(24,49,38,.08);
    background:linear-gradient(120deg,#fbfcfa 0%,#f4f7f3 58%,#eaf0ea 100%);
    box-shadow:0 18px 50px rgba(36,61,49,.06);
}
.hero-copy { position:relative; z-index:3; padding:50px 52px; max-width:750px; }
.hero-overline { color:#7c8a84; font-size:10px; text-transform:uppercase; letter-spacing:.17em; }
.hero-title { margin-top:13px; font-family:"Space Grotesk",sans-serif; font-size:clamp(45px,5vw,72px); font-weight:700; line-height:.93; letter-spacing:-.065em; color:#14221c; }
.hero-accent { color:#217355; }
.hero-tagline { margin-top:13px; color:#267254; font-size:18px; font-weight:600; }
.hero-description { max-width:650px; margin-top:10px; color:#65736d; font-size:14px; line-height:1.65; }
.hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
.hero-pill { padding:8px 11px; border:1px solid #dfe8e2; border-radius:10px; background:rgba(255,255,255,.68); color:#63716b; font-size:10px; }
.hero-pill strong { color:#2c6550; }
.globe { position:absolute; right:-20px; top:-48px; width:470px; height:470px; border-radius:50%; background:radial-gradient(circle at 32% 26%,#effff4 0,#bde4c9 11%,#5ba477 31%,#236747 57%,#0f3f30 78%,#092a21 100%); box-shadow:-24px 20px 65px rgba(26,78,54,.24), inset -28px -20px 65px rgba(0,0,0,.26); }
.globe:before { content:""; position:absolute; inset:18px; border-radius:50%; background:repeating-radial-gradient(circle,rgba(235,255,241,.10) 0 1px,transparent 2px 22px); opacity:.5; }
.globe:after { content:""; position:absolute; left:48px; top:82px; width:355px; height:204px; border:1px solid rgba(220,250,229,.33); border-radius:50%; transform:rotate(-18deg); box-shadow:0 0 0 30px rgba(221,250,230,.04),0 0 0 60px rgba(221,250,230,.025); }
.globe-dot { position:absolute; right:168px; top:151px; width:8px; height:8px; border-radius:50%; background:#d9ffe7; box-shadow:0 0 18px rgba(197,255,217,.9); animation:orbitPulse 2.7s infinite; }
@keyframes orbitPulse {0%,100%{transform:scale(.8);opacity:.65}50%{transform:scale(1.17);opacity:1}}

/* Cards */
.section-heading { margin-top:24px; font-family:"Space Grotesk",sans-serif; font-size:19px; font-weight:700; color:#1b2a23; letter-spacing:-.025em; }
.section-subheading { margin-top:3px; margin-bottom:14px; color:#7b8782; font-size:11px; }
.card { padding:18px; border:1px solid #dfe7e2; border-radius:18px; background:rgba(255,255,255,.88); box-shadow:0 8px 24px rgba(39,63,51,.035); }
.card-title { font-family:"Space Grotesk",sans-serif; font-size:15px; font-weight:700; color:#21332b; }
.card-text { margin-top:5px; color:#74807b; font-size:11px; line-height:1.5; }
.card-icon { width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:12px; background:#e7f3eb; color:#267455; font-size:17px; }
.mode-card { position:relative; padding:15px 18px; border:1px solid #dfe7e2; border-radius:15px; background:white; overflow:hidden; }
.mode-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:linear-gradient(180deg,#55aa7d,transparent); }
.mode-name { font-family:"Space Grotesk",sans-serif; color:#244136; font-weight:700; font-size:15px; }
.mode-description { margin-top:4px; color:#7a8781; font-size:10px; }

/* Inputs */
[data-testid="stFileUploader"] { border:1px dashed #c9d8cf; border-radius:15px; background:#fbfcfb; }
[data-baseweb="textarea"] { border:1px solid #dce5df !important; border-radius:14px !important; background:#fbfcfb !important; }
[data-baseweb="textarea"]:focus-within { border-color:#8cbea0 !important; box-shadow:0 0 0 3px rgba(80,148,105,.06) !important; }
textarea { color:#22342b !important; font-size:13px !important; }
[data-testid="stTextArea"] label { display:none; }

/* Buttons */
.stButton > button { border-radius:11px !important; border:1px solid #d5e2d9 !important; background:white !important; color:#285442 !important; font-weight:600 !important; transition:all .18s ease !important; }
.stButton > button:hover { transform:translateY(-1px); border-color:#91bba2 !important; box-shadow:0 8px 20px rgba(38,103,70,.08); }
.primary-action button { background:#26785a !important; color:white !important; border-color:#26785a !important; }

/* Metrics / outputs */
[data-testid="stMetric"] { min-height:88px; background:white; border:1px solid #dfe7e2; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#7b8781 !important; font-size:9px !important; text-transform:uppercase; letter-spacing:.09em; }
[data-testid="stMetricValue"] { color:#244a3a !important; font-family:"Space Grotesk",sans-serif; }
.result-panel { padding:20px 22px; border-radius:17px; border:1px solid #dfe7e2; background:white; animation:entry .35s ease-out; }
.result-answer { color:#30453b; font-size:14px; line-height:1.7; }
@keyframes entry {from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
.retrieval-card { padding:13px; margin-bottom:12px; border-radius:16px; border:1px solid #dfe7e2; background:white; transition:all .18s ease; }
.retrieval-card:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(38,77,58,.07); }
.rank-pill { display:inline-block; padding:4px 7px; border-radius:999px; background:#e7f3eb; color:#2c7758; font-size:9px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
.trace-card { padding:4px 14px; border-radius:15px; border:1px solid #dfe7e2; background:#fbfcfb; }
.trace-row { display:flex; gap:9px; padding:10px 0; border-bottom:1px solid #edf1ee; }
.trace-row:last-child { border-bottom:none; }
.trace-dot { width:7px; height:7px; flex-shrink:0; margin-top:5px; border-radius:50%; background:#58a97d; box-shadow:0 0 8px rgba(88,169,125,.34); }
.trace-title { color:#294139; font-size:11px; font-weight:700; }
.trace-status { color:#408c67; margin-left:7px; font-size:9px; }
.trace-message { margin-top:2px; color:#7d8984; font-size:10px; line-height:1.45; }
div[data-testid="stExpander"] { border-color:#dfe7e2 !important; border-radius:13px !important; background:rgba(255,255,255,.58) !important; }
[data-testid="stImage"] img { border-radius:12px; }
.footer { margin-top:38px; padding-top:15px; border-top:1px solid #dfe6e1; color:#8a958f; font-size:10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="logo-mark"></div><div><div class="brand-name">SatQuery</div><div class="brand-subtitle">Remote Sensing Intelligence</div></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="nav-caption">Workspace</div>',
    unsafe_allow_html=True,
)

for icon, page in [("⌂", "Home"), ("⌁", "Analysis"), ("◷", "History"), ("⚙", "Settings"), ("?", "Help")]:
    active = "nav-active" if st.session_state.page == page else ""
    st.sidebar.markdown(f'<div class="nav-wrap {active}">', unsafe_allow_html=True)
    if st.sidebar.button(
        f"{icon}  {page}",
        key=f"nav_{page}",
        use_container_width=True,
    ):
        st.session_state.page = page
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="sidebar-system"><small>SatQuery Core</small><div class="sidebar-online"><i></i>ONLINE</div></div>',
    unsafe_allow_html=True,
)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="hero"><div class="globe"></div><div class="globe-dot"></div><div class="hero-copy"><div class="hero-overline">Satellite Imagery · Real Insights</div><div class="hero-title">Sat<span class="hero-accent">Query</span></div><div class="hero-tagline">Explore the Earth. Find answers.</div><div class="hero-description">Upload satellite imagery, ask a natural-language question, and get meaningful insights about our changing planet.</div><div class="hero-pills"><div class="hero-pill"><strong>Multiple sensors</strong></div><div class="hero-pill"><strong>Rich analysis</strong></div><div class="hero-pill"><strong>Visual evidence</strong></div></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Start exploring</div><div class="section-subheading">Upload imagery and ask SatQuery what you want to understand.</div>',
        unsafe_allow_html=True,
    )

    upload_col, query_col = st.columns([1.0, 1.18], gap="large")

    with upload_col:
        st.markdown(
            '<div class="card"><div class="card-icon">↑</div><div class="card-title">Upload satellite image(s)</div><div class="card-text">PNG, JPG and TIFF/GeoTIFF inputs are supported.</div></div>',
            unsafe_allow_html=True,
        )
        home_uploads = st.file_uploader(
            "Home imagery",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="home_uploads",
            label_visibility="collapsed",
        )

    with query_col:
        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card-title">Ask a question</div><div class="card-text">Describe what you want to discover. SatQuery handles the workflow selection.</div>',
            unsafe_allow_html=True,
        )
        home_query = st.text_area(
            "Home question",
            value=st.session_state.query,
            placeholder=(
                "e.g. What are the buildings?\n"
                "What land-use changes are visible?\n"
                "Find satellite scenes showing vegetation."
            ),
            height=105,
            key="home_query_box",
            label_visibility="collapsed",
        )

        if st.button(
            "Analyze  →",
            key="home_analyze",
            type="primary",
            use_container_width=True,
        ):
            if home_query.strip():
                st.session_state.query = home_query.strip()
                saved = []
                for uploaded_file in home_uploads or []:
                    saved.append(save_uploaded_file(uploaded_file))
                st.session_state.pending_paths = saved
                st.session_state.page = "Analysis"
                st.rerun()
            else:
                st.error("Enter a question first.")

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading">What can you explore?</div><div class="section-subheading">SatQuery combines remote-sensing models into a single analysis experience.</div>',
        unsafe_allow_html=True,
    )

    features = [
        ("▥", "Identify", "Find buildings, roads, water bodies, vegetation and more."),
        ("▱", "Analyze", "Understand land use, terrain and environmental patterns."),
        ("◫", "Detect", "Discover temporal changes and extract key insights."),
        ("⌘", "Multi-sensor", "Work with optical and SAR imagery for broader understanding."),
    ]

    feature_cols = st.columns(4)
    for index, (icon, title, description) in enumerate(features):
        with feature_cols[index]:
            st.markdown(
                f'<div class="card"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-text">{description}</div></div>',
                unsafe_allow_html=True,
            )


# ============================================================
# ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Analysis":

    st.markdown(
        '<div class="section-heading">Analysis</div><div class="section-subheading">Choose a workflow or let SatQuery decide automatically.</div>',
        unsafe_allow_html=True,
    )

    analysis_mode = st.selectbox(
        "Workflow",
        [
            "Automatic",
            "Semantic Retrieval",
            "Optical + SAR",
            "Multitemporal Change",
            "Grounding / Localization",
            "Remote-Sensing VQA",
        ],
        key="analysis_mode",
    )

    mode_description = {
        "Automatic": "SatQuery selects the remote-sensing specialist from your query and inputs.",
        "Semantic Retrieval": "Search indexed satellite scenes using natural-language semantics.",
        "Optical + SAR": "Analyze complementary optical and SAR observations together.",
        "Multitemporal Change": "Detect and localize changes between two observations.",
        "Grounding / Localization": "Locate requested objects or regions inside satellite imagery.",
        "Remote-Sensing VQA": "Ask visual questions about satellite observations.",
    }[analysis_mode]

    st.markdown(
        f'<div class="mode-card"><div class="mode-name">{esc(analysis_mode)}</div><div class="mode-description">{esc(mode_description)}</div></div>',
        unsafe_allow_html=True,
    )

    optical_file = None
    sar_file = None
    uploaded_files = []
    use_demo = False

    if analysis_mode == "Semantic Retrieval":
        st.info(
            "Semantic Retrieval searches the indexed imagery. No upload is required."
        )

    elif analysis_mode == "Optical + SAR":
        a, b = st.columns(2)
        with a:
            optical_file = st.file_uploader(
                "Optical RGB GeoTIFF",
                type=["tif", "tiff"],
                key="analysis_optical",
            )
        with b:
            sar_file = st.file_uploader(
                "SAR GeoTIFF",
                type=["tif", "tiff"],
                key="analysis_sar",
            )

        with st.expander("Use verified BRIGHT demo pair"):
            if DEMO_OPTICAL.exists() and DEMO_SAR.exists():
                st.success("Verified Optical + SAR demo pair available.")
                use_demo = st.checkbox(
                    "Use demo pair",
                    key="use_demo_pair",
                )
            else:
                st.warning("Demo pair not found in the local project.")

    else:
        uploaded_files = st.file_uploader(
            "Satellite imagery",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="analysis_uploads",
        )

    query = st.text_area(
        "Mission query",
        value=st.session_state.query,
        placeholder=(
            "Ask SatQuery what you want to understand from the imagery..."
        ),
        height=120,
        key="analysis_query",
        label_visibility="collapsed",
    )

    quick_queries = [
        "Where are the buildings?",
        "What changed between these two images?",
        "Compare the optical and SAR imagery.",
        "Find satellite images of residential areas.",
    ]

    st.caption("Example queries")
    qcols = st.columns(2)
    for index, prompt in enumerate(quick_queries):
        with qcols[index % 2]:
            if st.button(
                prompt,
                key=f"analysis_quick_{index}",
                use_container_width=True,
            ):
                st.session_state.query = prompt
                st.rerun()

    if st.session_state.pending_paths:
        with st.expander("Imagery carried from Home", expanded=False):
            for path in st.session_state.pending_paths:
                st.code(path)

    if st.button(
        "Analyze  →",
        key="analysis_run",
        type="primary",
        use_container_width=True,
    ):

        if not query.strip():
            st.error("Please enter a query.")
            st.stop()

        image_paths: list[str] = []

        if analysis_mode == "Semantic Retrieval":
            image_paths = []

        elif analysis_mode == "Optical + SAR":
            if use_demo:
                image_paths = [str(DEMO_OPTICAL), str(DEMO_SAR)]
            else:
                if not optical_file or not sar_file:
                    st.error("Please upload both the Optical RGB and SAR imagery.")
                    st.stop()
                image_paths = [
                    save_uploaded_file(optical_file),
                    save_uploaded_file(sar_file),
                ]

        else:
            if uploaded_files:
                image_paths = [
                    save_uploaded_file(file)
                    for file in uploaded_files
                ]
            elif st.session_state.pending_paths:
                image_paths = list(st.session_state.pending_paths)
            elif analysis_mode == "Automatic" and looks_like_retrieval_query(query):
                image_paths = []
            else:
                st.error("Please upload at least one satellite image.")
                st.stop()

        effective_query = query.strip()

        if analysis_mode == "Optical + SAR":
            lowered = effective_query.lower()
            if "optical" not in lowered and "sar" not in lowered:
                effective_query = "Perform Optical and SAR analysis. " + effective_query

        elif analysis_mode == "Multitemporal Change":
            lowered = effective_query.lower()
            if "change" not in lowered and "changed" not in lowered:
                effective_query = "Perform multitemporal change analysis. " + effective_query

        elif analysis_mode == "Grounding / Localization":
            lowered = effective_query.lower()
            if not any(word in lowered for word in ["where", "locate", "highlight", "find", "identify"]):
                effective_query = "Locate and highlight " + effective_query

        try:
            with st.spinner("SatQuery is analyzing the Earth observation..."):
                response = requests.post(
                    f"{API_URL}/api/analyze",
                    json={"query": effective_query, "images": image_paths},
                    timeout=600,
                )

            if response.status_code != 200:
                st.error(f"Analysis failed ({response.status_code})")
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.code(str(detail))
                st.stop()

            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the SatQuery backend.")
            st.code("uvicorn backend.app.main:app --reload")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("The analysis timed out.")
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        st.session_state.query = query.strip()
        st.session_state.pending_paths = image_paths
        add_history(result)

        st.divider()
        st.markdown(
            '<div class="section-heading">SatQuery Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-panel"><div class="result-answer">{esc(result.get("answer", "No answer generated."))}</div></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Task", result.get("task", "—"))

        with m2:
            st.metric("Tool", result.get("tool", "—"))

        with m3:
            confidence = result.get("confidence")
            if confidence is None:
                value = "—"
            elif result.get("confidence_type") == "semantic_similarity":
                value = f"{float(confidence):.4f}"
            else:
                value = f"{float(confidence) * 100:.1f}%"
            st.metric("Confidence / Score", value)

        with m4:
            change = result.get("change_percentage")
            value = "—" if change is None else f"{float(change):.2f}%"
            st.metric("Changed", value)

        metadata = []
        if result.get("model"):
            metadata.append(f"Model: {result.get('model')}")
        if result.get("device"):
            metadata.append(f"Device: {result.get('device')}")
        if result.get("search_backend"):
            metadata.append(f"Search: {result.get('search_backend')}")
        if metadata:
            st.caption("  ·  ".join(metadata))

        evidence = result.get("evidence", {})

        # ----------------------------------------------------
        # Semantic Retrieval
        # ----------------------------------------------------
        if result.get("tool") == "semantic_retrieval":
            st.markdown(
                '<div class="section-heading">Scene Retrieval</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Ranked satellite scenes using RemoteCLIP semantic similarity."
            )
            retrieval_results = result.get(
                "results",
                evidence.get("retrieved_images", []),
            )
            if retrieval_results:
                for index, item in enumerate(retrieval_results, start=1):
                    if not isinstance(item, dict):
                        continue
                    image_path = item.get("path")
                    name = item.get(
                        "name",
                        Path(image_path).name if image_path else f"Scene {index}",
                    )
                    score = item.get("score")
                    st.markdown(
                        '<div class="retrieval-card">',
                        unsafe_allow_html=True,
                    )
                    left, right = st.columns([1, 1.6])
                    with left:
                        if image_path:
                            path = Path(image_path)
                            if path.exists():
                                st.image(
                                    str(path),
                                    caption=f"Rank {index}",
                                    use_container_width=True,
                                )
                            else:
                                st.warning("Retrieved image file not found.")
                    with right:
                        st.markdown(
                            f'<span class="rank-pill">Rank {index}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"### {esc(name)}")
                        if score is not None:
                            st.metric(
                                "Semantic Similarity",
                                f"{float(score):.4f}",
                            )
                        if image_path:
                            with st.expander("Source path"):
                                st.code(str(image_path))
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No matching scenes were found in the current retrieval index.")

        # ----------------------------------------------------
        # Grounding
        # ----------------------------------------------------
        if result.get("tool") == "grounding":
            st.markdown(
                '<div class="section-heading">Grounding Map</div>',
                unsafe_allow_html=True,
            )
            grounding_image = evidence.get("grounding_image")
            if grounding_image:
                path = Path(grounding_image)
                if path.exists():
                    st.image(
                        str(path),
                        caption="Text-guided region localization",
                        use_container_width=True,
                    )
            detections = result.get("detections", [])
            for index, detection in enumerate(detections, start=1):
                if not isinstance(detection, dict):
                    continue
                label = detection.get("label", detection.get("phrase", "object"))
                score = detection.get("score", detection.get("confidence"))
                box = detection.get("box", detection.get("bbox"))
                st.write(f"**Region {index}** — {label}")
                c1, c2 = st.columns(2)
                with c1:
                    if score is not None:
                        try:
                            st.write(f"Confidence: {float(score) * 100:.1f}%")
                        except Exception:
                            st.write(f"Confidence: {score}")
                with c2:
                    if box is not None:
                        st.write(f"Box: {box}")

        # ----------------------------------------------------
        # Optical + SAR
        # ----------------------------------------------------
        if result.get("tool") == "optical_sar":
            st.markdown(
                '<div class="section-heading">Optical–SAR Evidence</div>',
                unsafe_allow_html=True,
            )
            evidence_image = evidence.get("evidence_image")
            if evidence_image:
                path = Path(evidence_image)
                if path.exists():
                    st.image(
                        str(path),
                        caption="Optical + SAR analysis",
                        use_container_width=True,
                    )
            c1, c2, c3 = st.columns(3)
            with c1:
                value = result.get("changed_pixels")
                if value is not None:
                    st.metric("Changed Pixels", f"{int(value):,}")
            with c2:
                value = result.get("valid_pixels")
                if value is not None:
                    st.metric("Valid Pixels", f"{int(value):,}")
            with c3:
                value = result.get("changed_region_probability")
                if value is not None:
                    st.metric("Region Probability", f"{float(value):.3f}")

        # ----------------------------------------------------
        # Multitemporal
        # ----------------------------------------------------
        if result.get("tool") == "change_detection":
            st.markdown(
                '<div class="section-heading">Temporal Change Evidence</div>',
                unsafe_allow_html=True,
            )
            change_overlay = evidence.get("change_overlay")
            if change_overlay:
                path = Path(change_overlay)
                if path.exists():
                    st.image(
                        str(path),
                        caption="Detected temporal change",
                        use_container_width=True,
                    )
            regions = result.get("regions", [])
            for index, region in enumerate(regions, start=1):
                st.write(
                    f"**Region {index}:** "
                    f"x={region.get('x')} · y={region.get('y')} · "
                    f"width={region.get('width')} · height={region.get('height')}"
                )

        # ----------------------------------------------------
        # VQA / caption
        # ----------------------------------------------------
        if result.get("tool") == "vqa":
            st.markdown(
                '<div class="section-heading">Remote-Sensing Vision</div>',
                unsafe_allow_html=True,
            )
            source_image = evidence.get("image") or evidence.get("source_image")
            if source_image:
                path = Path(source_image)
                if path.exists():
                    st.image(
                        str(path),
                        caption="Source satellite observation",
                        use_container_width=True,
                    )

        # ----------------------------------------------------
        # Extra evidence / metadata
        # ----------------------------------------------------
        if result.get("confidence_note"):
            with st.expander("Reliability note"):
                st.write(result.get("confidence_note"))

        if result.get("validation"):
            with st.expander("Input validation"):
                st.json(result.get("validation"))

        if result.get("checkpoint_info"):
            with st.expander("Model information"):
                st.json(result.get("checkpoint_info"))

        render_trace(result.get("trace", []))

        with st.expander("Developer response"):
            st.json(result)


# ============================================================
# HISTORY PAGE
# ============================================================

elif st.session_state.page == "History":
    st.markdown(
        '<div class="section-heading">History</div><div class="section-subheading">Recent analyses from this browser session.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No analyses yet.")
    else:
        for index, item in enumerate(st.session_state.history, start=1):
            st.markdown(
                f'<div class="card"><span class="rank-pill">Analysis {index}</span><div class="card-title">{esc(item.get("query", ""))}</div><div class="card-text">{esc(item.get("task", ""))} · {esc(item.get("tool", ""))}</div></div>',
                unsafe_allow_html=True,
            )
            with st.expander("View result"):
                st.write(item.get("answer", ""))


# ============================================================
# SETTINGS PAGE
# ============================================================

elif st.session_state.page == "Settings":
    st.markdown(
        '<div class="section-heading">Settings</div><div class="section-subheading">Local prototype configuration.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><div class="card-icon">⚙</div><div class="card-title">SatQuery Core</div><div class="card-text">'
        f'Backend: {esc(API_URL)}<br>Interface: Streamlit<br>Agent orchestration: enabled'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HELP PAGE
# ============================================================

elif st.session_state.page == "Help":
    st.markdown(
        '<div class="section-heading">Help</div><div class="section-subheading">How to use SatQuery.</div>',
        unsafe_allow_html=True,
    )
    help_items = [
        ("Upload imagery", "Provide one image, a temporal pair, or an Optical + SAR pair depending on the workflow."),
        ("Ask naturally", "Describe what you need rather than naming the model you think should be used."),
        ("Analyze", "SatQuery validates inputs, chooses a workflow and executes the selected specialist."),
        ("Inspect evidence", "Review visual outputs, confidence, validation and the execution trace."),
    ]
    for title, description in help_items:
        st.markdown(
            f'<div class="card"><div class="card-title">{esc(title)}</div><div class="card-text">{esc(description)}</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer"><strong>SatQuery</strong> · Explore the Earth. Find answers.</div>',
    unsafe_allow_html=True,
)
