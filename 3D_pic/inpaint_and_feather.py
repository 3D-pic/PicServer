# inpaint_and_feather.py
import cv2, numpy as np
from pathlib import Path

ROOT = Path(r"D:\pycharm\3D_pic")
LAYER_DIR = ROOT / "output" / "layers"
OUT = ROOT / "output" / "layers_inpainted"
OUT.mkdir(parents=True, exist_ok=True)

def inpaint_and_feather(layer_path, radius=15, feather=15):
    rgba = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
    if rgba is None: return
    bgr = rgba[:,:,:3]
    alpha = rgba[:,:,3]
    hole_mask = (alpha==0).astype('uint8')*255
    if hole_mask.sum() > 0:
        inpainted = cv2.inpaint(bgr, hole_mask, radius, cv2.INPAINT_TELEA)
    else:
        inpainted = bgr
    # feather alpha
    alpha_blur = cv2.GaussianBlur(alpha.astype('float32'), (feather|1,feather|1), 0)
    alpha_blur = np.clip(alpha_blur, 0, 255).astype('uint8')
    out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
    out[:,:,3] = alpha_blur
    return out

if __name__ == "__main__":
    for p in sorted(LAYER_DIR.glob("layer_*.png")):
        out = inpaint_and_feather(p, radius=3, feather=51)
        if out is not None:
            cv2.imwrite(str(OUT / p.name), out)
            print("Processed", p.name)
