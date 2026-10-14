import os
import re
from typing import Tuple

import fitz
from pymongo.collection import Collection

from src import constant
from src.client.mongodb_config import MongoConfig
from src.fields.manual_images import ManualImages

manual_images_collection: Collection = MongoConfig.get_collection("manual_images")
image_save_dir = constant.image_save_dir
pdf_path = constant.pdf_path


def handle_image(img: Tuple, img_index: int, page: fitz.Page) -> ManualImages | None:
    """按 Polo 手册的图题版式提取图片及其关联标题。"""
    xref = img[0]
    base_image = page.parent.extract_image(xref)

    # 只排除极小的图标，保留仪表盘等有效插图。
    if base_image["width"] <= 34:
        return None

    image_path = save_image(base_image, img_index, page.number)
    rects = page.get_image_rects(xref)
    if not rects:
        return None
    img_rect = rects[0]

    # 将标题搜索范围限制在图片下方 50pt，减少正文中引用图号的干扰。
    search_distance = 50
    expanded_rect = fitz.Rect(
        img_rect.x0, img_rect.y1, img_rect.x1, img_rect.y1 + search_distance
    ).intersect(page.rect)

    title = None
    blocks = page.get_text("blocks")
    # 先检查靠近图片底部的文字块。
    blocks.sort(key=lambda block: block[1])
    for block in blocks:
        if not fitz.Rect(block[:4]).intersects(expanded_rect):
            continue
        text = block[4].strip()
        # 图题以“图＋数字”开头；正文中的“见图15”等引用不会命中。
        match = re.match(r"^图\s*([0-9]+)", text)
        if match:
            tail = text[match.end():].strip("：: \n")
            title = f"图{match.group(1)}" + (f" {tail}" if tail else "")
            break

    return ManualImages(image_path=image_path, page=page.number + 1, title=title)


def save_image(base_image: dict, img_index: int, page_number: int) -> str:
    """保存图片并返回文件路径；页号使用 PDF 中的实际页序。"""
    image_name = f"page{page_number + 1}_img{img_index + 1}.{base_image['ext']}"
    image_path = os.path.join(image_save_dir, image_name)
    os.makedirs(image_save_dir, exist_ok=True)
    with open(image_path, "wb") as output:
        output.write(base_image["image"])
    return image_path
