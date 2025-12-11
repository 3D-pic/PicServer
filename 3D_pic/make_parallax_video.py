# make_parallax_video.py
import argparse
from moviepy.editor import ImageClip, CompositeVideoClip
from pathlib import Path
import numpy as np
import sys
import cv2  # 添加cv2用于读取图像尺寸

# 获取当前脚本所在目录作为项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
FIXED_OUTPUT_DIR = PROJECT_ROOT / "output"


def calculate_video_size(image_path, max_height=1920, max_width=2560, target_aspect=None, width_reduction=0.2):
    """
    智能计算视频尺寸，视频宽度比图像宽度小一定比例

    Args:
        image_path: 输入图像路径
        max_height: 最大高度限制
        max_width: 最大宽度限制
        target_aspect: 目标宽高比（可选），如 "9:16" 或 "4:3"
        width_reduction: 宽度减少比例，默认0.2表示减少20%

    Returns:
        (width, height) 视频尺寸
    """
    try:
        # 读取图像获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}，使用默认尺寸")
            return 1440, 1920  # 默认竖屏尺寸

        img_h, img_w = img.shape[:2]
        img_ratio = img_w / img_h

        print(f"原始图像尺寸: {img_w}x{img_h}, 宽高比: {img_ratio:.2f}")

        # 如果有指定目标宽高比，则使用目标比例
        if target_aspect:
            if ":" in target_aspect:
                # 解析 "width:height" 格式
                w_ratio, h_ratio = map(float, target_aspect.split(":"))
                target_ratio = w_ratio / h_ratio
            else:
                # 尝试解析为浮点数
                try:
                    target_ratio = float(target_aspect)
                except:
                    target_ratio = img_ratio
        else:
            # 使用原始图像比例，但宽度减少一定比例
            target_ratio = img_ratio

        # 计算尺寸：让视频宽度比图像宽度小一定比例
        if target_ratio >= 1:  # 横屏或方屏
            # 以图像高度为基准，计算原始视频宽度
            h = min(max_height, int(max_width / target_ratio))
            w = int(h * target_ratio)

            # 应用宽度减少：让视频宽度比图像宽度小一定比例
            # 首先计算图像在视频高度下的宽度
            img_width_at_video_height = int(img_w * (h / img_h))
            # 减少宽度
            reduced_width = int(img_width_at_video_height * (1 - width_reduction))
            w = min(w, reduced_width)

            # 确保不超过最大宽度
            if w > max_width:
                w = max_width
                h = int(w / target_ratio)
        else:  # 竖屏
            # 以图像宽度为基准，计算原始视频高度
            w = min(max_width, int(max_height * target_ratio))
            h = int(w / target_ratio)

            # 对于竖屏图像，我们可能希望减少高度而不是宽度
            # 计算图像在视频宽度下的高度
            img_height_at_video_width = int(img_h * (w / img_w))
            # 减少高度
            reduced_height = int(img_height_at_video_width * (1 - width_reduction))
            h = min(h, reduced_height)

            # 确保不超过最大高度
            if h > max_height:
                h = max_height
                w = int(h * target_ratio)

        # 确保尺寸是偶数（视频编码要求）
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1

        # 确保最小尺寸
        w = max(w, 640)
        h = max(h, 480)

        print(f"计算出的视频尺寸: {w}x{h}, 宽高比: {w / h:.2f}")
        print(f"宽度减少比例: {width_reduction * 100:.0f}%")

        return w, h

    except Exception as e:
        print(f"计算视频尺寸失败，使用默认值: {e}")
        return 1440, 1920


def create_parallax_video(image_path: str, depth: int, parallax: int, duration: int,
                          camera_angle: int, output_path: Path,
                          target_height=1920, target_width=2560, aspect_ratio=None, width_reduction=0.2):
    """
    创建视差效果视频

    Args:
        image_path: 原始图像路径
        depth: 分层深度数
        parallax: 视差强度
        duration: 视频时长（秒）
        camera_angle: 相机角度
        output_path: 输出视频路径
        target_height: 目标高度
        target_width: 目标宽度
        aspect_ratio: 目标宽高比，如 "9:16" 或 "4:3"
        width_reduction: 宽度减少比例
    """
    # 视频参数
    fps = 30

    # 智能计算视频尺寸（应用宽度减少）
    w, h = calculate_video_size(
        image_path,
        max_height=target_height,
        max_width=target_width,
        target_aspect=aspect_ratio,
        width_reduction=width_reduction
    )

    # 确定分层目录
    layer_dir = FIXED_OUTPUT_DIR / "layers_fixed"

    if not layer_dir.exists():
        print(f"错误：分层目录不存在: {layer_dir}")
        print("请确保先运行 fix_all_layers.py")
        sys.exit(1)

    clips = []

    print("=== 创建视差效果视频 ===")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入图像: {image_path}")
    print(f"分层目录: {layer_dir}")
    print(f"输出目录: {FIXED_OUTPUT_DIR}")
    print(f"视频尺寸: {w}x{h}")
    print(f"帧率: {fps}")
    print(f"时长: {duration}秒")
    print(f"视差强度: {parallax}")
    print(f"相机角度: {camera_angle}度")

    # === 固定背景层 ===
    try:
        # 读取原始图像尺寸
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_h, img_w = img.shape[:2]
        img_ratio = img_w / img_h

        # 创建背景层 - 填充整个视频区域
        bg = ImageClip(str(image_path))

        # 计算背景缩放比例，使其填充整个视频区域
        # 视频宽高比
        video_ratio = w / h

        if video_ratio > img_ratio:
            # 视频比图像宽，需要裁剪上下部分
            bg = bg.resize(width=w)
            # 计算需要裁剪的高度
            scaled_height = int(bg.h * (w / bg.w))
            if scaled_height < h:
                # 如果缩放后高度不够，使用高度填充
                bg = bg.resize(height=h)
        else:
            # 视频比图像高，需要裁剪左右部分
            bg = bg.resize(height=h)
            # 计算需要裁剪的宽度
            scaled_width = int(bg.w * (h / bg.h))
            if scaled_width < w:
                # 如果缩放后宽度不够，使用宽度填充
                bg = bg.resize(width=w)

        # 设置位置居中
        bg = bg.set_position("center").set_duration(duration)
        clips.append(bg)  # 放在最底层
        print("✓ 添加背景层（填充模式）")

    except Exception as e:
        print(f"✗ 无法加载背景图像: {str(e)}")
        sys.exit(1)

    # === 分层图放在上面 ===
    layer_files = sorted(layer_dir.glob("layer_*.png"))
    n = len(layer_files)

    if n == 0:
        print(f"警告：在 {layer_dir} 中没有找到 layer_*.png 文件")
        # 尝试查找其他PNG文件
        layer_files = sorted(layer_dir.glob("*.png"))
        if n == 0:
            print("错误：没有找到任何PNG图层文件")
            sys.exit(1)

    print(f"找到 {n} 个图层文件")

    for i, p in enumerate(layer_files):
        try:
            clip = ImageClip(str(p)).set_duration(duration)

            # 调整图层尺寸以填充视频区域
            layer_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if layer_img is not None:
                layer_h, layer_w = layer_img.shape[:2]
                layer_ratio = layer_w / layer_h

                if video_ratio > layer_ratio:
                    # 视频比图层宽，需要裁剪上下部分
                    clip = clip.resize(width=w)
                else:
                    # 视频比图层高，需要裁剪左右部分
                    clip = clip.resize(height=h)
            else:
                # 如果无法读取图层，使用默认调整
                if w / h > clip.w / clip.h:
                    clip = clip.resize(width=w)
                else:
                    clip = clip.resize(height=h)

            # 根据视差强度和深度计算振幅
            # 使用提供的 parallax 参数，并根据层级调整
            base_amplitude = parallax
            # 振幅根据视频宽度动态调整，并考虑宽度减少
            amp = base_amplitude * (n - i) / n * (w / 1080)

            # 添加相机角度影响（将角度转换为弧度并调整运动）
            angle_rad = np.radians(camera_angle)

            def make_pos(t, amp=amp, angle=angle_rad):
                # 水平运动
                x = amp * np.sin(2 * np.pi * (t / duration) * 1.0)

                # 垂直运动（受相机角度影响）
                y = amp * 0.1 * np.sin(2 * np.pi * (t / duration) * 0.5 + angle)

                # 计算居中位置
                x_pos = x + (w - clip.w) / 2
                y_pos = y + (h - clip.h) / 2

                return (x_pos, y_pos)

            clip = clip.set_pos(make_pos)
            clips.append(clip)
            print(f"✓ 添加图层 {i + 1}/{n}: {p.name} (振幅: {amp:.1f})")

        except Exception as e:
            print(f"✗ 无法处理图层 {p.name}: {str(e)}")
            continue

    if len(clips) <= 1:  # 只有背景层，没有有效图层
        print("错误：没有有效的图层可以添加到视频中")
        sys.exit(1)

    # 创建合成视频
    print("合成视频中...")
    try:
        comp = CompositeVideoClip(clips, size=(w, h)).set_duration(duration)

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入视频文件
        print(f"正在写入视频文件: {output_path}")
        comp.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio=False,
            threads=4,
            preset="medium",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )

        print(f"✓ 视频生成完成: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"✗ 视频生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="创建视差效果视频：将分层图像合成为具有视差运动的视频")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--depth", type=int, default=5, help="分层深度数")
    parser.add_argument("--parallax", type=int, default=100, help="视差强度")
    parser.add_argument("--duration", type=int, default=15, help="视频时长（秒）")
    parser.add_argument("--camera-angle", type=int, default=30, help="相机角度（度）")
    parser.add_argument("--input-dir", type=str, default=None, help="输入分层目录（可选）")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（可选）")
    parser.add_argument("--width", type=int, default=None, help="视频宽度（可选，不指定则自动计算）")
    parser.add_argument("--height", type=int, default=None, help="视频高度（可选，不指定则自动计算）")
    parser.add_argument("--max-width", type=int, default=2560, help="最大视频宽度")
    parser.add_argument("--max-height", type=int, default=1920, help="最大视频高度")
    parser.add_argument("--aspect-ratio", type=str, default=None,
                        help="目标宽高比，如 '9:16', '4:3', '16:9'（可选）")
    parser.add_argument("--width-reduction", type=float, default=0.2,
                        help="宽度减少比例，0.2表示减少20%，默认0.2")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")

    args = parser.parse_args()

    try:
        # 解析输入图像路径
        input_path = Path(args.image)

        # 确定输入分层目录
        if args.input_dir:
            layer_dir = Path(args.input_dir)
        else:
            # 使用项目根目录下的固定目录
            layer_dir = FIXED_OUTPUT_DIR / "layers_fixed"

        # 确定输出视频路径
        if args.output:
            output_path = Path(args.output)
        else:
            # 使用项目根目录下的固定目录
            video_name = f"video_{args.duration}s.mp4"
            output_path = FIXED_OUTPUT_DIR / video_name

        # 检查输入文件是否存在
        if not input_path.exists():
            print(f"错误：输入图像不存在: {input_path}")
            sys.exit(1)

        # 如果用户指定了宽度和高度，使用指定值
        if args.width and args.height:
            w, h = args.width, args.height
            print(f"使用指定的视频尺寸: {w}x{h}")
            aspect_ratio = None
        else:
            w, h = None, None
            aspect_ratio = args.aspect_ratio

        # 创建视差视频
        video_path = create_parallax_video(
            image_path=str(input_path),
            depth=args.depth,
            parallax=args.parallax,
            duration=args.duration,
            camera_angle=args.camera_angle,
            output_path=output_path,
            target_height=args.max_height,
            target_width=args.max_width,
            aspect_ratio=aspect_ratio,
            width_reduction=args.width_reduction
        )

        print(f"视频文件已保存到: {video_path}")
        return video_path

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()