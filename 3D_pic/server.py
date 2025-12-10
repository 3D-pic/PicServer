from fastapi import FastAPI, UploadFile
import shutil
from pathlib import Path
from run_pipeline import run_pipeline
from fastapi.staticfiles import StaticFiles

app = FastAPI()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

# 如果没有 output 目录，创建一个
OUTPUT_DIR.mkdir(exist_ok=True)

# 静态视频目录
app.mount("/result", StaticFiles(directory=str(OUTPUT_DIR)), name="result")

@app.post("/upload_image/")
async def upload_image(file: UploadFile):
    upload_dir = ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    temp_path = upload_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    video_path = run_pipeline(str(temp_path))

    return {"video": f"/result/{video_path.name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
