from pathlib import Path
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline.load_image import load_image, preprocess_image

ROOT_DIR = Path(__file__).resolve().parents[1].parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Spine Image API")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".dcm"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    target_name = f"{uuid.uuid4().hex}{suffix}"
    target_path = UPLOAD_DIR / target_name

    contents = await file.read()
    target_path.write_bytes(contents)

    image = load_image(target_path)
    preprocessed = preprocess_image(image)

    return JSONResponse(
        {
            "filename": target_name,
            "path": str(target_path),
            "shape": list(preprocessed.shape),
        }
    )
