# inpaint_and_feather.py
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
FIXED_OUTPUT_DIR = PROJECT_ROOT / "output"


def inpaint_and_feather(layer_path: Path, radius: int = 15, feather: int = 15):
    """
    对图层进行修复和羽化处理

    Args:
        layer_path: 图层文件路径
        radius: 修复半径
        feather: 羽化半径

    Returns:
        修复和羽化后的图像数组 (BGRA格式)
    """
    # 读取RGBA图像
    rgba = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        print(f"警告：无法读取图像 {layer_path}")
        return None

    # 分离颜色和alpha通道
    bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # 创建空洞掩码（alpha为0的区域）
    hole_mask = (alpha == 0).astype('uint8') * 255

    # 如果有空洞，进行修复
    if hole_mask.sum() > 0:
        inpainted = cv2.inpaint(bgr, hole_mask, radius, cv2.INPAINT_TELEA)
    else:
        inpainted = bgr

    # 对alpha通道进行羽化
    # 确保羽化半径为奇数
    feather_size = feather if feather % 2 == 1 else feather + 1
    alpha_blur = cv2.GaussianBlur(alpha.astype('float32'),
                                  (feather_size, feather_size), 0)
    alpha_blur = np.clip(alpha_blur, 0, 255).astype('uint8')

    # 合并修复后的颜色和羽化后的alpha
    out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha_blur

    return out


def process_image(image_path: str, depth: int, parallax: int, duration: int, camera_angle: int,
                  radius: int = 3, feather: int = 51):
    """
    处理所有图层，进行修复和羽化

    Args:
        image_path: 输入图像路径
        depth: 分层深度数
        parallax: 视差强度
        duration: 视频时长
        camera_angle: 相机角度
        radius: 修复半径
        feather: 羽化半径

    Returns:
        处理后的图层目录路径
    """
    # 输入图层目录（来自make_layers.py）
    layer_dir = FIXED_OUTPUT_DIR / "layers"

    # 输出修复后的图层目录
    out_dir = FIXED_OUTPUT_DIR / "layers_inpainted"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not layer_dir.exists():
        print(f"错误：图层目录不存在: {layer_dir}")
        print("请确保先运行 make_layers.py")
        sys.exit(1)

    print("=== 图层修复和羽化处理 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入图层目录: {layer_dir}")
    print(f"输出目录: {out_dir}")
    print(f"修复半径: {radius}")
    print(f"羽化半径: {feather}")

    # 处理所有图层文件
    processed_count = 0
    layer_files = sorted(layer_dir.glob("layer_*.png"))

    if not layer_files:
        print(f"警告：在 {layer_dir} 中没有找到 layer_*.png 文件")
        return str(out_dir)

    for p in layer_files:
        try:
            # 进行修复和羽化处理
            out = inpaint_and_feather(p, radius=radius, feather=feather)
            if out is not None:
                # 保存处理后的图像
                output_path = out_dir / p.name
                cv2.imwrite(str(output_path), out)
                processed_count += 1
                print(f"✔ 处理完成: {p.name} -> {output_path.name}")
            else:
                print(f"✗ 处理失败: {p.name}")
        except Exception as e:
            print(f"✗ 处理错误 {p.name}: {str(e)}")

    print(f"=== 完成：处理了 {processed_count}/{len(layer_files)} 个图层 ===")

    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="图层修复和羽化：修复空洞并对边缘进行羽化处理")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--depth", type=int, default=5, help="分层深度数")
    parser.add_argument("--parallax", type=int, default=100, help="视差强度")
    parser.add_argument("--duration", type=int, default=15, help="视频时长")
    parser.add_argument("--camera-angle", type=int, default=30, help="相机角度")
    parser.add_argument("--radius", type=int, default=3, help="修复半径")
    parser.add_argument("--feather", type=int, default=51, help="羽化半径")
    parser.add_argument("--input-dir", type=str, default=None, help="输入图层目录（可选）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（可选）")

    args = parser.parse_args()

    try:
        # 如果指定了输入目录，使用指定目录
        if args.input_dir:
            layer_dir = Path(args.input_dir)
        else:
            # 否则使用项目根目录下的固定目录
            layer_dir = FIXED_OUTPUT_DIR / "layers"

        # 如果指定了输出目录，使用指定目录
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            # 否则使用项目根目录下的固定目录
            out_dir = FIXED_OUTPUT_DIR / "layers_inpainted"

        # 确保输出目录存在
        out_dir.mkdir(parents=True, exist_ok=True)

        if not layer_dir.exists():
            print(f"错误：图层目录不存在: {layer_dir}")
            print("请确保先运行 make_layers.py")
            sys.exit(1)

        print("=== 图层修复和羽化处理 ===")
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"输入图层目录: {layer_dir}")
        print(f"输出目录: {out_dir}")
        print(f"修复半径: {args.radius}")
        print(f"羽化半径: {args.feather}")

        # 处理所有图层文件
        processed_count = 0
        layer_files = sorted(layer_dir.glob("layer_*.png"))

        if not layer_files:
            print(f"警告：在 {layer_dir} 中没有找到 layer_*.png 文件")
            # 尝试查找其他PNG文件
            layer_files = sorted(layer_dir.glob("*.png"))
            if not layer_files:
                print("错误：没有找到任何PNG图层文件")
                sys.exit(1)

        for p in layer_files:
            try:
                # 进行修复和羽化处理
                out = inpaint_and_feather(p, radius=args.radius, feather=args.feather)
                if out is not None:
                    # 保存处理后的图像
                    output_path = out_dir / p.name
                    cv2.imwrite(str(output_path), out)
                    processed_count += 1
                    print(f"✔ 处理完成: {p.name} -> {output_path.name}")
                else:
                    print(f"✗ 处理失败: {p.name}")
            except Exception as e:
                print(f"✗ 处理错误 {p.name}: {str(e)}")

        print(f"=== 完成：处理了 {processed_count}/{len(layer_files)} 个图层 ===")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()