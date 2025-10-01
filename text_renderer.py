#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文文本渲染工具

OpenCV 的 cv2.putText 不支持中文，这里使用 Pillow 在图像上绘制中文，
并在内部完成 PIL(Image) 与 OpenCV(np.ndarray, BGR) 之间的转换。
"""

from typing import Tuple, Optional, List
import os
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # Pillow 未安装时，提供占位实现
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


def _candidate_font_paths() -> List[str]:
    """返回常见的中文字体路径候选列表（按优先级排序）。"""
    candidates: List[str] = []

    # Windows 常见中文字体
    candidates.extend([
        r"C:\\Windows\\Fonts\\msyh.ttc",      # 微软雅黑
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",   # 黑体
        r"C:\\Windows\\Fonts\\simsun.ttc",   # 宋体
        r"C:\\Windows\\Fonts\\Microsoft YaHei.ttf",
    ])

    # macOS 常见中文字体
    candidates.extend([
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti.ttf",
    ])

    # Linux 常见中文字体
    candidates.extend([
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ])

    # 备选：DejaVuSans（不保证支持中文，但比没有强）
    try:
        import matplotlib
        data_path = matplotlib.get_data_path()
        dejavu = os.path.join(data_path, "fonts", "ttf", "DejaVuSans.ttf")
        candidates.append(dejavu)
    except Exception:
        pass

    return candidates


def get_default_cn_font() -> Optional[str]:
    """
    返回一个可用的中文字体路径；如果未找到，返回 None。
    调用方可选择在 None 时退回到英文 putText。
    """
    for path in _candidate_font_paths():
        if os.path.exists(path):
            return path
    return None


def draw_text_cn(
    image_bgr: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 20,
    font_path: Optional[str] = None,
    anchor: str = "lt",
    stroke_width: int = 1,
    stroke_fill: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    在 BGR 图像上绘制中文文本。

    Args:
        image_bgr: OpenCV BGR 图像。
        text: 文本内容（支持中文）。
        position: (x, y) 左上角坐标（与 anchor 搭配）。
        color: 文本颜色（B, G, R）。
        font_size: 字号。
        font_path: 字体路径；为 None 时自动寻找常见中文字体。
        anchor: Pillow 文本锚点，默认左上角 'lt'。
        stroke_width: 描边宽度。
        stroke_fill: 描边颜色（B, G, R）。

    Returns:
        绘制后的新 BGR 图像（不会修改原图）。
    """

    if Image is None or ImageFont is None:  # Pillow 不可用，直接返回原图
        return image_bgr

    # 选择字体
    selected_font = font_path or get_default_cn_font()
    try:
        if selected_font is None:
            raise FileNotFoundError("No Chinese font found")
        font = ImageFont.truetype(selected_font, font_size)
    except Exception:
        # 找不到合适字体则直接返回（避免报错）
        return image_bgr

    # BGR -> RGB for Pillow
    image_rgb = image_bgr[:, :, ::-1].copy()
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Pillow 使用 RGB 颜色
    r, g, b = int(color[2]), int(color[1]), int(color[0])
    sr, sg, sb = int(stroke_fill[2]), int(stroke_fill[1]), int(stroke_fill[0])

    draw.text(
        position,
        text,
        fill=(r, g, b),
        font=font,
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=(sr, sg, sb),
    )

    # RGB -> BGR back to OpenCV (ensure C-contiguous and writeable)
    result_bgr = np.array(pil_img, copy=True)[:, :, ::-1].copy()
    # 标记为可写，避免后续 OpenCV 绘制时报只读错误
    try:
        result_bgr.setflags(write=True)
    except Exception:
        pass
    return result_bgr

