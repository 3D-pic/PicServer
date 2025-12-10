from moviepy.editor import ImageClip, CompositeVideoClip
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\pycharm\3D_pic")
LAYER_DIR = ROOT / "output" / "layers_fixed"
INPUT_IMAGE = ROOT / "input.jpg"     # 原图
OUT = ROOT / "output" / "video_15s.mp4"

fps = 30
duration = 15
w, h = 1080, 1920

clips = []

# === 新增：固定背景层 ===
bg = ImageClip(str(INPUT_IMAGE)).resize(height=h).set_position("center").set_duration(duration)
clips.append(bg)   # 放在最底层

# === 分层图放在上面 ===
layer_files = sorted(Path(LAYER_DIR).glob("layer_*.png"))
n = len(layer_files)

for i, p in enumerate(layer_files):
    clip = ImageClip(str(p)).set_duration(duration)
    clip = clip.resize(height=h)

    amp = (n - i) * 30

    def make_pos(t, amp=amp):
        x = amp * np.sin(2*np.pi*(t/duration)*1.0)
        y = 10 * np.sin(2*np.pi*(t/duration)*0.5)
        return (x + (w - clip.w)/2, y + (h - clip.h)/2)

    clip = clip.set_pos(make_pos)
    clips.append(clip)

comp = CompositeVideoClip(clips, size=(w, h)).set_duration(duration)
comp.write_videofile(str(OUT), fps=fps, codec="libx264", threads=4)
