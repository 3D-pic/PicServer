# make_layers.py
import cv2, numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

ROOT = Path(r"D:\pycharm\3D_pic")
OUT = ROOT / "output" / "layers"
OUT.mkdir(parents=True, exist_ok=True)

def depth_to_masks(depth, n_layers=5):
    h,w = depth.shape
    X = depth.reshape(-1,1)
    km = KMeans(n_clusters=n_layers, random_state=0, n_init=10).fit(X)
    labels = km.labels_.reshape(h,w)
    means = [depth[labels==i].mean() for i in range(n_layers)]
    order = np.argsort(means)  # nearest -> farthest if smaller depth = closer
    masks = []
    for i in range(n_layers):
        cls = order[i]
        mask = (labels == cls).astype('uint8') * 255
        masks.append(mask)
    return masks

def save_layers(img_path, depth_path, n_layers=5):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    depth = np.load(str(depth_path))
    masks = depth_to_masks(depth, n_layers=n_layers)
    for i, mask in enumerate(masks):
        # optional: smooth and expand mask edges
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_s = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        # create rgba
        bgr = img.copy()
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = mask_s
        outp = OUT / f"layer_{i:02d}.png"
        cv2.imwrite(str(outp), rgba)
        print("Saved", outp)

if __name__ == "__main__":
    save_layers(ROOT / "input.jpg", ROOT / "output" / "depth.npy", n_layers=5)
