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


def calculate_video_size_with_margin(image_path, max_amplitude, max_height=1920, max_width=2560,
                                     target_aspect=None, safety_margin=0.2):
    """
    计算考虑图层运动后的安全视频尺寸

    Args:
        image_path: 输入图像路径
        max_amplitude: 最大运动振幅
        max_height: 最大高度限制
        max_width: 最大宽度限制
        target_aspect: 目标宽高比（可选）
        safety_margin: 安全边距比例，默认0.2表示左右各留20%边距

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
        print(f"最大运动振幅: {max_amplitude}像素")

        # 计算需要保留的图像宽度（考虑运动范围）
        # 运动范围：左右各 max_amplitude 像素
        needed_image_width = img_w + 2 * max_amplitude

        # 应用安全边距：视频宽度 = 需要的图像宽度 × (1 - 2×safety_margin)
        video_width = int(needed_image_width * (1 - 2 * safety_margin))
        video_height = int(video_width / img_ratio) if img_ratio > 0 else img_h

        print(f"考虑运动后的图像宽度: {needed_image_width}")
        print(f"安全边距: {safety_margin * 100:.0f}%")
        print(f"初步视频尺寸: {video_width}x{video_height}")

        # 如果有指定目标宽高比，则调整
        if target_aspect:
            if ":" in target_aspect:
                w_ratio, h_ratio = map(float, target_aspect.split(":"))
                target_ratio = w_ratio / h_ratio
            else:
                try:
                    target_ratio = float(target_aspect)
                except:
                    target_ratio = video_width / video_height

            # 按目标比例调整
            if target_ratio >= 1:  # 横屏
                h = min(video_height, max_height, int(max_width / target_ratio))
                w = int(h * target_ratio)
            else:  # 竖屏
                w = min(video_width, max_width, int(max_height * target_ratio))
                h = int(w / target_ratio)
        else:
            w, h = video_width, video_height

        # 确保不超过最大尺寸
        if w > max_width:
            w = max_width
            h = int(w / (video_width / video_height))
        if h > max_height:
            h = max_height
            w = int(h * (video_width / video_height))

        # 确保尺寸是偶数（视频编码要求）
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1

        # 确保最小尺寸
        w = max(w, 640)
        h = max(h, 480)

        print(f"最终视频尺寸: {w}x{h}, 宽高比: {w / h:.2f}")
        print(f"相当于考虑运动后宽度的 {w / needed_image_width * 100:.1f}%")

        return w, h

    except Exception as e:
        print(f"计算视频尺寸失败，使用默认值: {e}")
        return 1440, 1920


def create_parallax_video(image_path: str, depth: int, parallax: int, duration: int,
                          camera_angle: int, output_path: Path,
                          target_height=1920, target_width=2560, aspect_ratio=None,
                          safety_margin=0.2, crop_final=True):
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
        safety_margin: 安全边距比例
        crop_final: 是否最终裁剪视频
    """
    # 视频参数
    fps = 30

    # 确定分层目录
    layer_dir = FIXED_OUTPUT_DIR / "layers_fixed"

    if not layer_dir.exists():
        print(f"错误：分层目录不存在: {layer_dir}")
        print("请确保先运行 fix_all_layers.py")
        sys.exit(1)

    # === 首先分析图层运动范围 ===
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

    # 计算最大运动振幅
    max_amplitude = 0
    for i in range(n):
        base_amplitude = parallax
        amp = base_amplitude * (n - i) / n
        max_amplitude = max(max_amplitude, abs(amp))

    print(f"计算的最大运动振幅: {max_amplitude:.1f}像素")

    # 智能计算视频尺寸（考虑运动范围）
    w, h = calculate_video_size_with_margin(
        image_path,
        max_amplitude=max_amplitude,
        max_height=target_height,
        max_width=target_width,
        target_aspect=aspect_ratio,
        safety_margin=safety_margin
    )

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
    print(f"安全边距: {safety_margin * 100:.0f}%")

    # === 固定背景层 ===
    try:
        # 读取原始图像尺寸
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        img_h, img_w = img.shape[:2]
        img_ratio = img_w / img_h

        # 创建背景层 - 使用原始完整尺寸
        bg = ImageClip(str(image_path))

        # 调整背景尺寸以包含运动范围
        # 计算需要的背景尺寸（原始图像 + 运动范围）
        needed_bg_width = img_w + 2 * max_amplitude
        needed_bg_height = int(needed_bg_width / img_ratio) if img_ratio > 0 else img_h

        # 调整背景尺寸
        if needed_bg_width / needed_bg_height > img_ratio:
            bg = bg.resize(width=needed_bg_width)
        else:
            bg = bg.resize(height=needed_bg_height)

        bg = bg.set_duration(duration)
        clips.append(bg)  # 放在最底层
        print(f"✓ 添加背景层 (尺寸: {bg.w}x{bg.h})")

    except Exception as e:
        print(f"✗ 无法加载背景图像: {str(e)}")
        sys.exit(1)

    # === 分层图放在上面 ===
    for i, p in enumerate(layer_files):
        try:
            # 读取图层
            layer_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if layer_img is None:
                print(f"✗ 无法读取图层: {p.name}")
                continue

            layer_h, layer_w = layer_img.shape[:2]
            layer_ratio = layer_w / layer_h

            # 创建图层剪辑
            clip = ImageClip(str(p)).set_duration(duration)

            # 调整图层尺寸以匹配背景（使用原始尺寸）
            clip = clip.resize((layer_w, layer_h))

            # 根据视差强度和深度计算振幅
            base_amplitude = parallax
            amp = base_amplitude * (n - i) / n

            # 添加相机角度影响（将角度转换为弧度并调整运动）
            angle_rad = np.radians(camera_angle)

            def make_pos(t, amp=amp, angle=angle_rad, bg_width=bg.w, bg_height=bg.h):
                # 水平运动
                x = amp * np.sin(2 * np.pi * (t / duration) * 1.0)

                # 垂直运动（受相机角度影响）
                y = amp * 0.1 * np.sin(2 * np.pi * (t / duration) * 0.5 + angle)

                # 计算在背景上的位置（居中）
                x_pos = x + (bg_width - clip.w) / 2
                y_pos = y + (bg_height - clip.h) / 2

                return (x_pos, y_pos)

            clip = clip.set_pos(make_pos)
            clips.append(clip)
            print(f"✓ 添加图层 {i + 1}/{n}: {p.name} (振幅: {amp:.1f}, 尺寸: {clip.w}x{clip.h})")

        except Exception as e:
            print(f"✗ 无法处理图层 {p.name}: {str(e)}")
            continue

    if len(clips) <= 1:  # 只有背景层，没有有效图层
        print("错误：没有有效的图层可以添加到视频中")
        sys.exit(1)

    # 创建合成视频（使用包含运动范围的尺寸）
    print(f"合成视频中 (包含运动范围)...")

    # 计算包含运动范围的合成尺寸
    composite_width = int(bg.w)
    composite_height = int(bg.h)

    # 计算裁剪区域（居中裁剪掉安全边距）
    if crop_final:
        # 计算裁剪区域（居中）
        crop_x = int((composite_width - w) / 2)
        crop_y = int((composite_height - h) / 2)

        print(f"最终裁剪区域: [{crop_x}:{crop_x + w}, {crop_y}:{crop_y + h}]")

        # 创建大尺寸合成视频，然后裁剪
        temp_comp = CompositeVideoClip(clips, size=(composite_width, composite_height)).set_duration(duration)

        # 裁剪到目标尺寸
        comp = temp_comp.crop(x1=crop_x, y1=crop_y, width=w, height=h)
    else:
        # 不裁剪，直接使用包含运动范围的尺寸
        comp = CompositeVideoClip(clips, size=(composite_width, composite_height)).set_duration(duration)
        w, h = composite_width, composite_height

    try:
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
    parser.add_argument("--safety-margin", type=float, default=0.2,
                        help="安全边距比例，0.2表示左右各留20%边距，默认0.2")
    parser.add_argument("--no-crop", action="store_true",
                        help="不裁剪最终视频，保留完整运动范围")
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
            safety_margin=args.safety_margin,
            crop_final=not args.no_crop
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