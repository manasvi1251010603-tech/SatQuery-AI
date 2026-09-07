# SatQuery-AI
🛰️ SatQuery AI

An Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis Through Natural-Language Queries

SatQuery AI is an AI-powered remote-sensing analysis platform that allows users to interact with satellite imagery using natural-language queries.

Instead of requiring users to manually select different GIS tools, preprocessing pipelines, and machine-learning models, SatQuery uses an agent-driven architecture to interpret a query and route it to the appropriate remote-sensing analysis workflow.

The system is designed around the idea:

Ask a question about the Earth. SatQuery selects the analysis required to answer it.

🚀 Project Overview

Satellite imagery analysis often requires expertise in:

Remote sensing

GIS software

Image preprocessing

Optical imagery

SAR imagery

Change detection

Object localization

Machine-learning models

Geospatial interpretation

SatQuery AI aims to simplify this workflow through a natural-language interface while retaining the underlying remote-sensing-specific processing.

The system currently combines multiple specialist capabilities:

                    Natural-Language Query
                              │
                              ▼
                    ┌──────────────────┐
                    │  Query Planner   │
                    │  / Task Router   │
                    └─────────┬────────┘
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
        VQA              Grounding          Change Detection
          │                   │                    │
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Optical + SAR    │
                    │     Analysis      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Semantic Retrieval│
                    └─────────┬─────────┘
                              │
                              ▼
                     Evidence + Trace

✨ Current Capabilities

1. Remote-Sensing Visual Question Answering

SatQuery can answer natural-language questions about satellite imagery using a remote-sensing-adapted Vision-Language Model.

Current model:

AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct

Example query:

What type of land cover is visible?

The VQA workflow:

Satellite Image
      ↓
Remote-Sensing VLM
      ↓
Natural-Language Answer

The source image is retained as visual evidence.

2. Text-Guided Region Grounding

SatQuery can locate objects or regions specified through natural language.

Example:

Where are the buildings?

or:

Highlight the roads.

The system extracts the requested target and sends it to the grounding specialist.

Current specialist:

Grounding DINO

Workflow:

Natural-Language Query
        ↓
Target Extraction
        ↓
Grounding DINO
        ↓
Detected Regions
        ↓
Bounding Boxes / Visualization

The backend returns:

Detected regions

Labels

Confidence scores

Bounding boxes

Grounding visualization

3. Multi-Temporal Change Detection

SatQuery supports comparison between two observations from different points in time.

Example:

What changed between these two images?

Current change-detection model:

AdaptFormer
deepang/adaptformer-LEVIR-CD

Workflow:

Before Image
      +
After Image
      ↓
Temporal Change Model
      ↓
Change Mask
      ↓
Change Overlay
      ↓
Changed Regions

The system returns:

Change mask

Change overlay

Changed regions

Change percentage

Model information

Execution trace

Important note

The current AdaptFormer implementation was tested successfully on its official LEVIR-CD example data.

A Sentinel-2 pair tested separately did not produce meaningful change predictions, demonstrating the importance of dataset/model compatibility.

4. Optical + SAR Multimodal Analysis

SatQuery integrates a specialist Optical–SAR change detection workflow.

This allows complementary information from:

Optical Earth-observation imagery

SAR imagery

to be analyzed jointly.

Current specialist:

GalaxEye EfficientNet-B0 U-Net

The implementation is based on the open-source:

third_party/galaxeye-eo-sar

The implemented architecture uses:

Input:
3-channel Optical RGB
+
1-channel SAR

Derived SAR representations:
- CLAHE
- Log representation

Total model input:
5 channels

Workflow:

Optical RGB ───────┐
                   │
                   ▼
             Modality-Aware
              Preprocessing
                   │
SAR ───────────────┘
                   ↓
        EfficientNet-B0 U-Net
                   ↓
          Change Probability
                   ↓
             Change Mask
                   ↓
            Evidence Panel

The backend performs validation of:

Band count

Image dimensions

CRS

Geospatial transform

Spatial bounds

The system generates:

Change mask GeoTIFF

Change probability GeoTIFF

Evidence visualization

Changed pixel count

Valid pixel count

Mean probabilities

Confidence indicator

5. Semantic Retrieval

SatQuery now supports natural-language semantic search over satellite imagery.

Current model:

RemoteCLIP ViT-B-32

Official RemoteCLIP checkpoint:

models/remoteclip/RemoteCLIP-ViT-B-32.pt

The system converts both images and natural-language queries into embeddings.

Workflow:

Natural-Language Query
          ↓
RemoteCLIP Text Encoder
          ↓
Semantic Embedding
          ↓
Cosine Similarity
          ↓
Ranked Satellite Scenes

Example:

Find satellite images of residential areas with buildings.

The retrieval system returns:

Image path

Image name

Semantic similarity score

Ranked results

Current retrieval backend

The project uses a lightweight NumPy vector index:

embeddings.npy
metadata.npy

instead of FAISS in the active retrieval process.

This was intentionally chosen because FAISS and the current Apple Silicon PyTorch/OpenCLIP runtime produced an OpenMP native-runtime conflict.

The retrieval functionality itself remains genuine vector-based semantic retrieval using RemoteCLIP embeddings.

6. Agentic Task Planning

SatQuery includes a task-planning layer that interprets a natural-language request and selects a specialist workflow.

Current planner:

backend/app/agent/planner.py

Current supported tasks:

VQA
CAPTION
GROUNDING
MULTITEMPORAL_CHANGE
OPTICAL_SAR
SEMANTIC_RETRIEVAL

The planner considers:

Query wording

Number of supplied images

Optical/SAR terminology

Temporal terminology

Grounding terminology

Retrieval terminology

Scene-description terminology

Example:

"Where are the buildings?"
        ↓
GROUNDING

"What changed between these two dates?"
        ↓
MULTITEMPORAL_CHANGE

"Compare the optical and SAR imagery."
        ↓
OPTICAL_SAR

"Find satellite images of residential areas."
        ↓
SEMANTIC_RETRIEVAL

7. Observable Execution Trace

SatQuery records an execution trace for the selected workflow.

Example:

Input Validation
       ↓
Query Planning
       ↓
Model Selection
       ↓
Preprocessing
       ↓
Inference
       ↓
Evidence Generation
       ↓
Confidence Estimation
       ↓
Response Generation

The trace is returned by the backend and displayed in the frontend.

This is intended to make the system more auditable and explainable than a generic chatbot response.

🧠 Architecture

Current high-level architecture:

                         USER
                          │
                          ▼
                Natural-Language Query
                          │
                          ▼
                 FastAPI Backend
                          │
                          ▼
                Agent / Task Planner
                          │
            ┌─────────────┼──────────────┐
            │             │              │
            ▼             ▼              ▼
          VQA         Grounding     Change Detection
            │             │              │
            │             │              │
            └─────────────┼──────────────┘
                          │
                          ▼
                    Optical + SAR
                          │
                          ▼
                  Semantic Retrieval
                          │
                          ▼
              Evidence + Confidence
                          │
                          ▼
                 Execution Trace
                          │
                          ▼
                    Streamlit UI

🗂️ Project Structure

satquery-ai/
│
├── backend/
│   ├── __init__.py
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── api/
│       │   ├── upload.py
│       │   └── analyze.py
│       │
│       ├── agent/
│       │   ├── planner.py
│       │   ├── router.py
│       │   └── trace.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── change_detection.py
│           ├── grounding.py
│           ├── optical_sar.py
│           ├── retrieval_bridge.py
│           ├── semantic_retrieval.py
│           └── vqa.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │
│   └── demo/
│       ├── temporal/
│       ├── adaptformer/
│       ├── bright/
│       ├── results/
│       ├── uploads/
│       └── retrieval_index/
│
├── frontend/
│   └── app.py
│
├── models/
│   └── remoteclip/
│
├── notebooks/
│
├── scripts/
│   ├── create_test_geotiff.py
│   ├── read_geotiff.py
│   ├── inspect_geotiff.py
│   ├── visualize_geotiff.py
│   ├── visualize_rgb.py
│   ├── create_rgb.py
│   ├── create_false_color.py
│   ├── search_sentinel2.py
│   ├── download_sentinel2.py
│   ├── adaptformer_test.py
│   ├── create_change_overlay.py
│   ├── create_change_report.py
│   ├── test_change_service.py
│   ├── test_planner.py
│   ├── test_vqa.py
│   ├── test_vqa_service.py
│   ├── test_grounding.py
│   └── test_remoteclip.py
│
├── third_party/
│   └── galaxeye-eo-sar/
│
├── docs/
│
├── .gitignore
├── README.md
└── ...

⚙️ Technology Stack

Backend

Python
FastAPI
Uvicorn
Pydantic

AI / ML

PyTorch
Transformers
OpenCLIP
RemoteCLIP
Grounding DINO
AdaptFormer
Segmentation Models PyTorch

Remote Sensing / Geospatial

Rasterio
NumPy
OpenCV
GeoTIFF
CRS-aware processing

Frontend

Streamlit

Development

VS Code
Git
GitHub
macOS / Apple Silicon

🛰️ Data and Models

SatQuery currently integrates several open-source/research models rather than training every model from scratch.

Remote-sensing VQA

AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct

Semantic Retrieval

RemoteCLIP ViT-B-32

Change Detection

deepang/adaptformer-LEVIR-CD

Optical + SAR

GalaxEye EfficientNet-B0 U-Net

Grounding

Grounding DINO

🧪 Current Testing Status

The following components have been tested during development:

Component

Status

GeoTIFF loading

✅

Remote-sensing VQA

✅

Grounding

✅

Multitemporal Change

✅

Optical + SAR

✅

RemoteCLIP checkpoint loading

✅

RemoteCLIP image-text retrieval

✅

Semantic Retrieval index creation

✅

Semantic Retrieval backend bridge

✅

Agent planner routing

✅

FastAPI analysis integration

✅

Streamlit integration

✅

🔍 Example Queries

VQA

What type of land cover is visible?

Grounding

Where are the buildings?

Highlight the roads.

Multitemporal Change

What changed between these two observations?

Optical + SAR

Compare the optical and SAR imagery and identify changed regions.

Semantic Retrieval

Find satellite images of residential areas with buildings.

Search for dense urban scenes.

🔐 Environment Separation

Different model groups are isolated into different Python environments because their dependency requirements can conflict.

Current environments include:

.venv/

Main SatQuery environment.

.venv-vqa/

Remote-sensing VQA environment.

.venv-retrieval/

RemoteCLIP semantic retrieval environment.

The retrieval backend uses:

retrieval_bridge.py

to execute the RemoteCLIP workflow inside the dedicated environment.

This avoids dependency conflicts with the main remote-sensing models.

▶️ Running SatQuery

1. Activate the main environment

source .venv/bin/activate

2. Start the backend

uvicorn backend.app.main:app --reload

3. Open another terminal

source .venv/bin/activate

4. Start the frontend

streamlit run frontend/app.py

🔄 Current Query Execution Flow

For an image-based query:

User Query
    ↓
FastAPI
    ↓
Planner
    ↓
Task Selection
    ↓
Required Input Validation
    ↓
Specialist Model
    ↓
Evidence Generation
    ↓
Confidence
    ↓
Execution Trace
    ↓
Response

For semantic retrieval:

User Query
    ↓
Planner
    ↓
SEMANTIC_RETRIEVAL
    ↓
Retrieval Bridge
    ↓
.venv-retrieval
    ↓
RemoteCLIP
    ↓
Text Embedding
    ↓
Cosine Similarity
    ↓
Ranked Satellite Scenes
    ↓
Evidence + Trace

🎯 Current Novelty Direction

SatQuery is being developed around several complementary capabilities:

1. Agentic Model Orchestration

A natural-language request is used to select the appropriate remote-sensing specialist workflow.

2. Semantic Retrieval

Natural-language descriptions can be mapped into a remote-sensing image embedding space to retrieve relevant scenes.

3. Task-Aware Remote-Sensing Processing

Different sensor and task types use different processing pipelines rather than treating every image as an ordinary photograph.

4. Multi-Temporal Change Analysis

The system can analyze observations across different points in time.

5. Optical–SAR Multimodal Analysis

Optical and SAR imagery can be processed together within a unified query-driven workflow.

6. Evidence-Grounded Outputs

Analysis results are accompanied by visual/model evidence and an observable execution trace.

⚠️ Current Limitations

SatQuery is currently a prototype and has several limitations.

Semantic Retrieval Corpus

The retrieval index currently contains only a small demonstration corpus.

A larger satellite-image corpus is required for meaningful retrieval evaluation.

Change Detection Generalization

Some pretrained change-detection models are dataset-specific. Model/data compatibility must therefore be considered when evaluating new satellite imagery.

Confidence

Current confidence values are workflow-specific indicators and should not be interpreted as universally calibrated probabilities.

VQA Reliability

Remote-sensing VLM outputs still require evidence-based verification. SatQuery therefore emphasizes visual evidence and execution traces rather than treating generated language as ground truth.

Geospatial Measurements

Advanced geospatial measurements and richer map interaction are planned for future iterations.

🚧 Roadmap

The next development stage focuses on turning the existing specialist modules into a more complete agentic remote-sensing system.

Phase 1 — Agentic Intelligence

Complex query understanding

Query decomposition

Multi-step task planning

Dynamic tool selection

Input compatibility validation

Multi-model execution graphs

Phase 2 — Geospatial Intelligence

Area measurement

Distance measurement

Coordinate extraction

Bounding-region measurements

CRS-aware geospatial calculations

Phase 3 — Interactive Mapping

Interactive satellite map

Grounding overlays

Change layers

Optical/SAR layers

Measurement layers

Region selection

Phase 4 — Multispectral Analysis

Band selection

NDVI

NDWI

NDBI

Spectral analysis tools

Natural-language spectral queries

Phase 5 — Explainability

Evidence fusion

Confidence reasoning

Source-region highlighting

Model outputs

Analysis provenance

Phase 6 — Intelligence Reports

Generate an automated report containing:

Query

Input imagery

Analysis performed

Key findings

Measurements

Maps

Change evidence

Grounding evidence

Model information

Confidence

Limitations

🏗️ Future Target Architecture

The long-term SatQuery architecture is planned as:

                       USER
                         │
                         ▼
              ┌────────────────────┐
              │ Query Understanding │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Agent Orchestrator │
              └─────────┬──────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
    Retrieval       Geo Tools       Specialist Models
        │               │                │
        │               │        ┌───────┼─────────┐
        │               │        │       │         │
        │               │        ▼       ▼         ▼
        │               │       VQA  Grounding  Change
        │               │
        │               │
        └───────────────┼────────────────┘
                        ▼
               Evidence Fusion
                        │
                        ▼
               Confidence / Reliability
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Interactive Map        Intelligence
                               Report

📌 Project Status

SatQuery AI currently has a functioning prototype containing:

✅ Remote-Sensing VQA
✅ Text-Guided Grounding
✅ Multi-Temporal Change Detection
✅ Optical + SAR Analysis
✅ Semantic Retrieval
✅ Agentic Task Routing
✅ Input Validation
✅ Evidence Generation
✅ Confidence Indicators
✅ Observable Execution Trace
✅ FastAPI Backend
✅ Streamlit Frontend

The project is now moving from individual specialist workflows toward a unified agentic geospatial intelligence system.

👥 Development

SatQuery AI is being developed as a Smart India Hackathon project.

The architecture emphasizes:

Remote Sensing
+
Artificial Intelligence
+
Vision-Language Models
+
Multimodal Analysis
+
Geospatial Intelligence
+
Agentic Orchestration

📄 License

Add the appropriate license for the final project and verify the licenses of all third-party models, datasets, and repositories before public distribution.

🙌 Acknowledgements

This project builds upon open-source and research work in:

Remote-sensing vision-language models

RemoteCLIP

Grounding DINO

AdaptFormer

GalaxEye EO-SAR change detection

Open-source Earth-observation datasets

All third-party models and repositories should be used in accordance with their respective licenses and attribution requirements.
