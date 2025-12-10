from pathlib import Path
from PIL import Image
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
LAYER_DIR = ROOT / "output" / "layers_inpainted"
OUT_DIR = ROOT / "output" / "layers_fixed"
OUT_DIR.mkdir(exist_ok=True)

# 小洞阈值（像素）；越大修补越多
SMALL_HOLE_SIZE = 500

def fix_small_holes(img: np.ndarray) -> np.ndarray:
    alpha = img[..., 3]
    rgb = img[..., :3]

    # 找透明区域
    mask = (alpha < 255).astype(np.uint8)

    # 统计连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    fixed = rgb.copy()
    new_alpha = alpha.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < SMALL_HOLE_SIZE:
            # 小洞 → 修复
            component_mask = (labels == i).astype(np.uint8) * 255
            repaired = cv2.inpaint(rgb, component_mask, 3, cv2.INPAINT_TELEA)
            fixed = np.where(component_mask[..., None] == 255, repaired, fixed)
            new_alpha[labels == i] = 255  # 修复后设为不透明
        else:
            # 大透明区域 → 保留
            pass

    return np.dstack([fixed, new_alpha])

def main():
    print("=== 只修复透明边缘，不覆盖大透明洞 ===")
    for path in sorted(LAYER_DIR.glob("*.png")):
        im = Image.open(path).convert("RGBA")
        arr = np.array(im)

        out = fix_small_holes(arr)
        Image.fromarray(out).save(OUT_DIR / path.name)
        print("✔ 修复:", path.name)

    print("=== 完成 ===")

if __name__ == "__main__":
    main()
