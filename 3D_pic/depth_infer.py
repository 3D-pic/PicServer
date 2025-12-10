# depth_infer.py
import os, cv2, torch, numpy as np
from pathlib import Path

# paths
ROOT = Path(r"D:\pycharm\3D_pic")
IMG_IN = ROOT / "input.jpg"      # 把你的 test.jpg/3D_pic/test.jpg 重命名为 input.jpg 或改这个路径
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# model selection
MODEL = "DPT_Large"  # or "DPT_Hybrid", "MiDaS_small"

print("Loading model:", MODEL)
midas = torch.hub.load("intel-isl/MiDaS", MODEL, trust_repo=True)
midas.to("cpu")
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
# choose transform (robust)
if MODEL in ("DPT_Large", "DPT_Hybrid"):
    transform = transforms.dpt_transform
else:
    transform = transforms.small_transform

def predict_depth(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transform(img_rgb)
    # ensure shape (N,C,H,W)
    if isinstance(t, torch.Tensor):
        if t.ndim == 3: input_tensor = t.unsqueeze(0)
        elif t.ndim == 4: input_tensor = t
        else: raise RuntimeError("Unexpected transform ndim: %d" % t.ndim)
    else:
        t = torch.from_numpy(np.array(t))
        input_tensor = t.unsqueeze(0) if t.ndim == 3 else t
    with torch.no_grad():
        prediction = midas(input_tensor)
    # normalize prediction to HxW
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        pred = prediction.squeeze(1)
    else:
        pred = prediction
    if pred.ndim == 3:
        pred = pred[0]
    pred = torch.nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0),
                                           size=img_rgb.shape[:2],
                                           mode="bicubic", align_corners=False).squeeze()
    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

if __name__ == "__main__":
    img = cv2.imread(str(IMG_IN))
    if img is None:
        raise SystemExit("Can't read input image: " + str(IMG_IN))
    depth = predict_depth(img)
    np.save(OUT_DIR / "depth.npy", depth)
    cv2.imwrite(str(OUT_DIR / "depth_vis.png"), (depth * 255).astype("uint8"))
    print("Saved depth to:", OUT_DIR)
