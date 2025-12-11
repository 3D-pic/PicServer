import subprocess
import sys
from pathlib import Path

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
PY = sys.executable


def run_pipeline(image_path: str, depth: int, parallax: int, duration: int, camera_angle: int):
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

    # 确保输出目录存在
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"=== 立体书生成管道开始 ===")
    print(f"项目根目录: {ROOT}")
    print(f"输出目录: {output_dir}")
    print(f"输入图像: {image_path}")
    print(f"参数: depth={depth}, parallax={parallax}, duration={duration}, camera_angle={camera_angle}")

    for step in steps:
        print(f"\n=== 正在执行: {step} ===")
        try:
            subprocess.run([
                PY, str(ROOT / step),
                "--image", image_path,
                "--depth", str(depth),
                "--parallax", str(parallax),
                "--duration", str(duration),
                "--camera-angle", str(camera_angle),
            ], check=True, cwd=ROOT)  # 设置工作目录为项目根目录
        except subprocess.CalledProcessError as e:
            print(f"✗ 步骤 {step} 执行失败，错误码: {e.returncode}")
            print(f"请检查 {step} 的详细错误信息")
            raise
        except Exception as e:
            print(f"✗ 步骤 {step} 执行异常: {str(e)}")
            raise

    # 视频文件路径（基于项目根目录）
    video_name = f"video_{duration}s.mp4"
    video_path = output_dir / video_name

    print(f"\n=== 管道执行完成 ===")
    print(f"期望视频路径: {video_path}")
    print(f"文件存在: {video_path.exists()}")

    if video_path.exists():
        print(f"✅ 视频生成成功: {video_path}")
        # 显示文件大小
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"文件大小: {size_mb:.2f} MB")
    else:
        print(f"⚠️ 视频文件未找到，检查输出目录: {output_dir}")
        # 列出输出目录中的所有文件
        if output_dir.exists():
            print(f"输出目录中的文件:")
            for file in output_dir.iterdir():
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    print(f"  - {file.name} ({size_kb:.1f} KB)")

    return video_path