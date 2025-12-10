import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\pycharm\3D_pic")
PY = sys.executable  # 当前 venv 的 Python，会自动保持正确环境

def run_pipeline() -> Path:
    """
    执行立体书生成全流程：
    1. 深度估计
    2. 分层 mask 生成
    3. 修补 + feather
    4. 图片修复
    5. 视差视频生成

    返回生成的视频文件路径
    """
    steps = [
        "depth_infer.py",
        "make_layers.py",
        "inpaint_and_feather.py",
        "fix_all_layers.py",
        "make_parallax_video.py"
    ]

    for step in steps:
        print(f"\n==== Running: {step} ====\n")
        subprocess.run([PY, str(ROOT / step)], check=True)

    # 假设视频输出固定在 ROOT / "output" / "parallax_video.mp4"
    video_path = ROOT / "output" / "video_15s.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"视频生成失败: {video_path}")
    print(f"\n>>> ALL DONE ✓\nGenerated video: {video_path}")
    return video_path

# 后端可以这样调用：
# video_file = run_pipeline()
