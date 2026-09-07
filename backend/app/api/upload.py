from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    suffix = Path(file.filename).suffix.lower()

    allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    file_id = uuid4().hex
    output_path = UPLOAD_DIR / f"{file_id}{suffix}"

    content = await file.read()
    output_path.write_bytes(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(output_path),
    }