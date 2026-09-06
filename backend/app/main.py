from fastapi import FastAPI

app = FastAPI(
    title="SatQuery AI",
    description="Agentic remote-sensing intelligence platform",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "SatQuery AI",
        "version": "0.1.0",
    }