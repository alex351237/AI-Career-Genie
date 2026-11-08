"""
Author  :Alex Aug
Date    :2026/11/8 17:01
describe: pdf文档加载器.
"""

from io import BytesIO
from typing import Iterator
import cv2
# import pymupdf
import fitz
import numpy as np
from PIL import Image
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from tqdm import tqdm

from backend.config import config
from thy_ocr import get_ocr_instance

IMAGE_AREA_RATIO_THRESHOLD = 0.1
IMAGE_WIDTH_THRESHOLD, IMAGE_HEIGHT_THRESHOLD = 64, 64


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """图像旋转修正"""
    # 获得图像尺寸和中心点
    h, w = image.shape[:2]
    rotate_center = (w / 2, h / 2)
    # 生成基础旋转矩阵（2x3 旋转矩阵）
    m = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    # 计算旋转后防止裁剪的新画布宽高
    new_w = int(h * np.abs(m[0, 1]) + w * np.abs(m[0, 0]))
    new_h = int(h * np.abs(m[0, 0]) + w * np.abs(m[1, 1]))
    # 平移修正（原来的平移偏移量加上单侧需要补充的宽度和高度）
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    # 执行仿射变换
    return cv2.warpAffine(image, m, (new_w, new_h))


class OCRPDFLoader(BaseLoader):
    """PDF 文档加载器"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        text_content = self.pdf2text()
        yield Document(page_content=text_content, metadata={'source': self.file_path})

    def pdf2text(self) -> str:
        pdf_doc = fitz.open(self.file_path)
        text_buffer = []

        ocr_obj = get_ocr_instance()
        try:
            pro_bar = tqdm(total=pdf_doc.page_count, desc='OCRPDFLoader processing')
            for page in pdf_doc:
                # 提取页面文字
                page_text = page.get_text('text')
                if page_text and page_text.strip():
                    text_buffer.append(page_text.strip())

                # 提取并处理页面嵌入图片
                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    # 每个图都有 xref 属性（唯一标识）
                    xref = img.get('xref')
                    if not xref:
                        continue

                    # 四元组：0, 1 是左上角的横纵坐标，2, 3 是右下角的横纵坐标
                    bbox = img['bbox']
                    # 获取图像的宽高并计算出面积
                    image_width, image_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    image_area = image_width * image_height
                    # 计算图像占页面的比例
                    page_area = page.rect.width * page.rect.height
                    area_ratio = image_area / page_area

                    # 通过预先设置的阈值确认图像是否需要做 OCR
                    if (area_ratio < IMAGE_AREA_RATIO_THRESHOLD or
                            image_width < IMAGE_WIDTH_THRESHOLD or
                            image_height < IMAGE_HEIGHT_THRESHOLD):
                        continue

                    # 提取图像 pixmap
                    pix = fitz.Pixmap(pdf_doc, xref)
                    try:
                        # 计算出色彩通道数（3 - RGB 彩色图或者 4 - CMYK 印刷图）
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes('png')
                        else:
                            # 带有 CMYK 等特殊色彩空间的转换处理
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix_rgb.tobytes('png')
                            pix_rgb = None  # 显式清除 C++ 底层引用防内存泄露

                        # 通过获取到的图像的二进制数据创建 PILImage 对象
                        image = Image.open(BytesIO(img_data)).convert('RGB')
                        # 将 PILImage 对象处理成 NumPy 数组（ndarray）
                        img_array = np.array(image)

                        # 旋转校正
                        if int(page.rotation) != 0:
                            img_array = _rotate_image(img_array, 360 - page.rotation)

                        # 执行 OCR 识别
                        result, _ = ocr_obj(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            text_buffer.append('\n'.join(ocr_result))
                    finally:
                        pix = None  # 显式清除 C++ 底层引用防内存泄露

                pro_bar.update(1)
            pro_bar.close()
        finally:
            pdf_doc.close()

        return '\n\n'.join(text_buffer)