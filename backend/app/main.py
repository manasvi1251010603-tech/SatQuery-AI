from fastapi import FastAPI

from backend.app.api.upload import router as upload_router

app = FastAPI(
    title="SatQuery AI",
    version="0.1.0",
)

app.include_router(upload_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "SatQuery AI",
    }
from backend.app.api.analyze import router as analyze_router

app.include_router(analyze_router)