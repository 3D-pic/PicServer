from fastapi import FastAPI, UploadFile, Form
import shutil
from pathlib import Path
from run_pipeline import run_pipeline
from fastapi.staticfiles import StaticFiles

app = FastAPI()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/result", StaticFiles(directory=str(OUTPUT_DIR)), name="result")

@app.post("/upload_image/")
async def upload_image(
    file: UploadFile,
    depth: int = Form(...),
    parallax: int = Form(...),
    duration: int = Form(...),
    camera_angle: int = Form(...)
):
    upload_dir = ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    temp_path = upload_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ★ 将参数传给 pipeline
    video_path = run_pipeline(
        image_path=str(temp_path),
        depth=depth,
        parallax=parallax,
        duration=duration,
        camera_angle=camera_angle
    )

    return {"video": f"/result/{video_path.name}"}
