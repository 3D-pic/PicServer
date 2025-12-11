# depth_infer.py
import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

# 模型选择
MODEL = "DPT_Large"  # or "DPT_Hybrid", "MiDaS_small"

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
FIXED_OUTPUT_DIR = PROJECT_ROOT / "output"


def setup_model():
    """加载深度估计模型"""
    print("Loading model:", MODEL)
    midas = torch.hub.load("intel-isl/MiDaS", MODEL, trust_repo=True)
    midas.to("cpu")
    midas.eval()

    # 加载对应的transform
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if MODEL in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform


def predict_depth(img_bgr, midas, transform):
    """预测深度图"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transform(img_rgb)

    # 确保张量形状为 (N, C, H, W)
    if isinstance(t, torch.Tensor):
        if t.ndim == 3:
            input_tensor = t.unsqueeze(0)
        elif t.ndim == 4:
            input_tensor = t
        else:
            raise RuntimeError(f"Unexpected transform ndim: {t.ndim}")
    else:
        t = torch.from_numpy(np.array(t))
        input_tensor = t.unsqueeze(0) if t.ndim == 3 else t

    # 进行预测
    with torch.no_grad():
        prediction = midas(input_tensor)

    # 处理预测结果
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        pred = prediction.squeeze(1)
    else:
        pred = prediction

    if pred.ndim == 3:
        pred = pred[0]

    # 插值到原始图像大小
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

    # 归一化深度图
    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    return depth


def main(image_path, output_dir=None):
    """主函数：处理单张图像并生成深度图"""
    # 使用项目根目录下的output文件夹，如果未指定则使用默认目录
    if output_dir is None:
        output_dir = FIXED_OUTPUT_DIR

    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查输入图像
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 加载图像
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    print(f"Processing image: {img_path.name}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print(f"Output directory: {output_dir}")

    # 加载模型并进行预测
    midas, transform = setup_model()
    depth = predict_depth(img, midas, transform)

    # 保存结果
    depth_npy_path = output_dir / "depth.npy"
    depth_vis_path = output_dir / "depth_vis.png"

    np.save(str(depth_npy_path), depth)
    cv2.imwrite(str(depth_vis_path), (depth * 255).astype("uint8"))

    print(f"Depth map saved to: {depth_npy_path}")
    print(f"Depth visualization saved to: {depth_vis_path}")

    return str(depth_npy_path), str(depth_vis_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深度估计：从单张图像生成深度图")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--depth", type=int, default=5, help="分层深度数（管道参数，此处不使用）")
    parser.add_argument("--parallax", type=int, default=100, help="视差强度（管道参数，此处不使用）")
    parser.add_argument("--duration", type=int, default=15, help="视频时长（管道参数，此处不使用）")
    parser.add_argument("--camera-angle", type=int, default=30, help="相机角度（管道参数，此处不使用）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（可选）")

    args = parser.parse_args()

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        # 使用项目根目录下的output文件夹
        output_dir = FIXED_OUTPUT_DIR

    try:
        depth_npy_path, depth_vis_path = main(args.image, output_dir)
        print(f"Successfully generated depth map: {depth_npy_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)