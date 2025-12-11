from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
from run_pipeline import run_pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ★ 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段直接全部放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 静态目录：可直接访问视频
app.mount("/result", StaticFiles(directory=str(OUTPUT_DIR)), name="result")


# ★ 视频下载接口（使用 Query）
@app.get("/download_video/")
def download_video(filename: str = Query(...)):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return {"error": "文件不存在"}

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=file_path.name
    )


# ★ 图片上传 → 运行 pipeline → 返回视频下载路径
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

    # ★ 运行你的 3D pipeline
    video_path = run_pipeline(
        image_path=str(temp_path),
        depth=depth,
        parallax=parallax,
        duration=duration,
        camera_angle=camera_angle
    )

    # 返回静态访问地址
    return {"video": f"/result/{video_path.name}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
