# make_layers.py
import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from pathlib import Path
import sys

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
FIXED_OUTPUT_DIR = PROJECT_ROOT / "output"


def depth_to_masks(depth: np.ndarray, n_layers: int = 5):
    """
    将深度图转换为分层掩码

    Args:
        depth: 深度图数组
        n_layers: 分层数量

    Returns:
        分层掩码列表（从近到远）
    """
    h, w = depth.shape
    X = depth.reshape(-1, 1)

    # 使用KMeans聚类进行深度分层
    km = KMeans(n_clusters=n_layers, random_state=0, n_init=10)
    labels = km.fit_predict(X).reshape(h, w)

    # 计算每个聚类的平均深度
    means = [depth[labels == i].mean() for i in range(n_layers)]

    # 按深度排序（从小到大，深度值越小表示距离越近）
    order = np.argsort(means)

    masks = []
    for i in range(n_layers):
        cls = order[i]
        mask = (labels == cls).astype('uint8') * 255
        masks.append(mask)

    return masks


def save_layers(img_path: Path, depth_path: Path, output_dir: Path, n_layers: int = 5,
                morph_size: int = 5):
    """
    保存分层图像

    Args:
        img_path: 原始图像路径
        depth_path: 深度图路径
        output_dir: 输出目录
        n_layers: 分层数量
        morph_size: 形态学操作核大小
    """
    # 读取原始图像
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 读取深度图
    depth = np.load(str(depth_path))

    # 确保深度图与图像大小一致
    if depth.shape != img.shape[:2]:
        print(f"警告：深度图大小 {depth.shape} 与图像大小 {img.shape[:2]} 不一致")
        # 调整深度图大小
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]))

    # 生成分层掩码
    masks = depth_to_masks(depth, n_layers=n_layers)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个分层
    for i, mask in enumerate(masks):
        # 使用形态学操作平滑掩码边缘
        if morph_size > 0:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            mask_s = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        else:
            mask_s = mask

        # 创建RGBA图像
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask_s

        # 保存分层图像
        output_path = output_dir / f"layer_{i:02d}.png"
        cv2.imwrite(str(output_path), rgba)
        print(f"保存分层 {i + 1}/{n_layers}: {output_path.name} (大小: {mask_s.shape})")

    return output_dir


def process_image(image_path: str, depth: int, parallax: int, duration: int, camera_angle: int):
    """
    处理图像，生成分层

    Args:
        image_path: 输入图像路径
        depth: 分层深度数
        parallax: 视差强度
        duration: 视频时长
        camera_angle: 相机角度

    Returns:
        分层图像目录路径
    """
    # 深度图路径（来自depth_infer.py）
    depth_path = FIXED_OUTPUT_DIR / "depth.npy"

    # 分层输出目录
    layers_dir = FIXED_OUTPUT_DIR / "layers"

    if not depth_path.exists():
        print(f"错误：深度图文件不存在: {depth_path}")
        print("请确保先运行 depth_infer.py")
        sys.exit(1)

    print("=== 生成分层图像 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入图像: {image_path}")
    print(f"深度图: {depth_path}")
    print(f"输出目录: {layers_dir}")
    print(f"分层数量: {depth} 层")

    # 生成并保存分层
    save_layers(
        img_path=Path(image_path),
        depth_path=depth_path,
        output_dir=layers_dir,
        n_layers=depth,
        morph_size=5
    )

    print(f"=== 完成：生成了 {depth} 个分层 ===")

    return str(layers_dir)


def main():
    parser = argparse.ArgumentParser(description="生成分层图像：根据深度图将图像分成多个图层")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--depth", type=int, default=5, help="分层深度数")
    parser.add_argument("--parallax", type=int, default=100, help="视差强度")
    parser.add_argument("--duration", type=int, default=15, help="视频时长")
    parser.add_argument("--camera-angle", type=int, default=30, help="相机角度")
    parser.add_argument("--depth-file", type=str, default=None, help="深度图文件路径（可选）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（可选）")
    parser.add_argument("--morph-size", type=int, default=5, help="形态学操作核大小")

    args = parser.parse_args()

    try:
        # 解析输入图像路径
        input_path = Path(args.image)

        # 确定深度图路径
        if args.depth_file:
            depth_path = Path(args.depth_file)
        else:
            # 使用项目根目录下的固定目录
            depth_path = FIXED_OUTPUT_DIR / "depth.npy"

        # 确定输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # 使用项目根目录下的固定目录
            output_dir = FIXED_OUTPUT_DIR / "layers"

        # 检查输入文件是否存在
        if not input_path.exists():
            print(f"错误：输入图像不存在: {input_path}")
            sys.exit(1)

        if not depth_path.exists():
            print(f"错误：深度图文件不存在: {depth_path}")
            print("请确保先运行 depth_infer.py")
            sys.exit(1)

        print("=== 生成分层图像 ===")
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"输入图像: {input_path}")
        print(f"深度图: {depth_path}")
        print(f"输出目录: {output_dir}")
        print(f"分层数量: {args.depth} 层")
        print(f"形态学核大小: {args.morph_size}")

        # 生成并保存分层
        save_layers(
            img_path=input_path,
            depth_path=depth_path,
            output_dir=output_dir,
            n_layers=args.depth,
            morph_size=args.morph_size
        )

        print(f"=== 完成：生成了 {args.depth} 个分层 ===")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()