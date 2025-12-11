# fix_all_layers.py
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import sys

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
FIXED_OUTPUT_DIR = PROJECT_ROOT / "output"


def fix_small_holes(img: np.ndarray, small_hole_size: int = 500) -> np.ndarray:
    """
    修复小透明洞，保留大透明区域

    Args:
        img: RGBA格式的图像数组
        small_hole_size: 小洞阈值（像素），越大修补越多

    Returns:
        修复后的RGBA图像数组
    """
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
        if area < small_hole_size:
            # 小洞 → 修复
            component_mask = (labels == i).astype(np.uint8) * 255
            repaired = cv2.inpaint(rgb, component_mask, 3, cv2.INPAINT_TELEA)
            fixed = np.where(component_mask[..., None] == 255, repaired, fixed)
            new_alpha[labels == i] = 255  # 修复后设为不透明
        else:
            # 大透明区域 → 保留
            pass

    return np.dstack([fixed, new_alpha])


def process_image(image_path: str, depth: int, parallax: int, duration: int, camera_angle: int):
    """
    处理单张图像，修复分层中的小洞

    Args:
        image_path: 输入图像路径
        depth: 分层深度数
        parallax: 视差强度
        duration: 视频时长
        camera_angle: 相机角度

    Returns:
        修复后的图层目录路径
    """
    # 输入图层目录（来自上一步骤）
    layer_dir = FIXED_OUTPUT_DIR / "layers_inpainted"

    # 输出修复后的图层目录
    out_dir = FIXED_OUTPUT_DIR / "layers_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not layer_dir.exists():
        print(f"错误：输入图层目录不存在: {layer_dir}")
        print("请确保先运行 inpaint_and_feather.py")
        sys.exit(1)

    print("=== 修复透明边缘，不覆盖大透明洞 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入图层目录: {layer_dir}")
    print(f"输出目录: {out_dir}")
    print(f"小洞阈值: 500像素")

    # 处理所有图层
    processed_count = 0
    for path in sorted(layer_dir.glob("*.png")):
        try:
            # 加载图像
            im = Image.open(path).convert("RGBA")
            arr = np.array(im)

            # 修复小洞
            out = fix_small_holes(arr, small_hole_size=500)

            # 保存修复后的图像
            output_path = out_dir / path.name
            Image.fromarray(out).save(output_path)

            processed_count += 1
            print(f"✔ 修复: {path.name} -> {output_path.name}")
        except Exception as e:
            print(f"✗ 处理失败 {path.name}: {str(e)}")

    if processed_count == 0:
        print("警告：没有找到任何PNG图层文件")
    else:
        print(f"=== 完成：修复了 {processed_count} 个图层 ===")

    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="修复分层图像：修复小透明洞，保留大透明区域")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--depth", type=int, default=5, help="分层深度数")
    parser.add_argument("--parallax", type=int, default=100, help="视差强度")
    parser.add_argument("--duration", type=int, default=15, help="视频时长")
    parser.add_argument("--camera-angle", type=int, default=30, help="相机角度")
    parser.add_argument("--hole-size", type=int, default=500, help="小洞阈值（像素）")
    parser.add_argument("--input-dir", type=str, default=None, help="输入图层目录（可选）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（可选）")

    args = parser.parse_args()

    try:
        # 如果指定了输入目录，使用指定目录
        if args.input_dir:
            layer_dir = Path(args.input_dir)
        else:
            # 否则使用项目根目录下的固定目录
            layer_dir = FIXED_OUTPUT_DIR / "layers_inpainted"

        # 如果指定了输出目录，使用指定目录
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            # 否则使用项目根目录下的固定目录
            out_dir = FIXED_OUTPUT_DIR / "layers_fixed"

        # 确保目录存在
        out_dir.mkdir(parents=True, exist_ok=True)

        if not layer_dir.exists():
            print(f"错误：输入图层目录不存在: {layer_dir}")
            print("请确保先运行 inpaint_and_feather.py")
            sys.exit(1)

        print("=== 修复透明边缘，不覆盖大透明洞 ===")
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"输入图层目录: {layer_dir}")
        print(f"输出目录: {out_dir}")
        print(f"小洞阈值: {args.hole_size}像素")

        # 处理所有图层
        processed_count = 0
        for path in sorted(layer_dir.glob("*.png")):
            try:
                # 加载图像
                im = Image.open(path).convert("RGBA")
                arr = np.array(im)

                # 修复小洞
                out = fix_small_holes(arr, small_hole_size=args.hole_size)

                # 保存修复后的图像
                output_path = out_dir / path.name
                Image.fromarray(out).save(output_path)

                processed_count += 1
                print(f"✔ 修复: {path.name} -> {output_path.name}")
            except Exception as e:
                print(f"✗ 处理失败 {path.name}: {str(e)}")

        if processed_count == 0:
            print("警告：没有找到任何PNG图层文件")
        else:
            print(f"=== 完成：修复了 {processed_count} 个图层 ===")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()