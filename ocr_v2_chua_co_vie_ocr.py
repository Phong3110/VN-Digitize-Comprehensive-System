"""
OCR Pipeline: Scanned PDF → Searchable PDF/A + JSON
Dự án: VN-Digitize - Module 1 & 2

Yêu cầu cài đặt:
    pip install pymupdf paddleocr reportlab tqdm

Sử dụng:
    python ocr_pipeline.py --input scan.pdf --output output_folder --dpi 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. KHỞI TẠO PADDLEOCR (1 lần duy nhất)
# ─────────────────────────────────────────────
def init_ocr(device: str = "gpu") -> PaddleOCR:
    print(f"[OCR] Khởi tạo PaddleOCR trên {device.upper()}...")
    ocr = PaddleOCR(
        # Module 1: Image Pre-processing
        use_doc_orientation_classify=True,   # Tự động xoay tài liệu 0/90/180/270°
        use_doc_unwarping=True,              # Nắn thẳng ảnh bị cong (scan sách, chụp nghiêng)
        use_textline_orientation=True,        # Phát hiện dòng chữ bị lật ngược

        # Module 2: OCR Engine
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        device=device,

        # Tăng độ nhạy detect chữ nhỏ
        text_det_limit_side_len=960,
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=2.0,
    )
    print("[OCR] Sẵn sàng.\n")
    return ocr


# ─────────────────────────────────────────────
# 2. CONVERT PDF → IMAGES (dùng pymupdf)
# ─────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[tuple[np.ndarray, tuple]]:
    """
    Trả về list các (numpy_image, (width_pt, height_pt))
    width_pt, height_pt là kích thước trang PDF gốc tính bằng points (1pt = 1/72 inch)
    """
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    print(f"[PDF] Đang đọc {len(doc)} trang từ '{pdf_path}' (DPI={dpi})...")
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        page_size_pt = (page.rect.width, page.rect.height)  # points
        pages.append((img, page_size_pt))

    doc.close()
    return pages


# ─────────────────────────────────────────────
# 3. CHẠY OCR TỪNG TRANG
# ─────────────────────────────────────────────
def ocr_page(ocr: PaddleOCR, img: np.ndarray) -> list[dict]:
    """
    Trả về list các text block:
    {
        "text": str,
        "confidence": float,
        "bbox": {"x1":int, "y1":int, "x2":int, "y2":int},
        "polygon": [[x,y], ...]   # 4 điểm polygon gốc
    }
    """
    results = ocr.predict(img)
    blocks = []

    for res in results:
        if res is None:
            continue

        polys   = res.get("rec_polys", [])
        texts   = res.get("rec_texts", [])
        scores  = res.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            if not text.strip():
                continue

            # Polygon gốc (4 điểm)
            polygon = poly.tolist() if hasattr(poly, "tolist") else list(poly)

            # Bounding box hình chữ nhật bao quanh polygon
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            blocks.append({
                "text": text,
                "confidence": round(float(score), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "polygon": polygon,
            })

    return blocks


# ─────────────────────────────────────────────
# 4. TẠO SEARCHABLE PDF/A (2 lớp)
# ─────────────────────────────────────────────
def register_vietnamese_font(font_dir: str = None):
    """
    Đăng ký font hỗ trợ tiếng Việt với ReportLab.
    Tìm font trong thư mục chỉ định hoặc font hệ thống Windows.
    """
    search_dirs = []
    if font_dir:
        search_dirs.append(font_dir)

    # Windows system fonts
    search_dirs += [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
    ]

    font_candidates = ["Arial", "times", "calibri", "verdana", "tahoma"]
    font_files = {
        "Arial":    ["arial.ttf", "Arial.ttf"],
        "times":    ["times.ttf", "Times.ttf", "timesnewroman.ttf"],
        "calibri":  ["calibri.ttf", "Calibri.ttf"],
        "verdana":  ["verdana.ttf", "Verdana.ttf"],
        "tahoma":   ["tahoma.ttf", "Tahoma.ttf"],
    }

    for font_name in font_candidates:
        for font_file in font_files[font_name]:
            for d in search_dirs:
                full_path = os.path.join(d, font_file)
                if os.path.exists(full_path):
                    try:
                        pdfmetrics.registerFont(TTFont("VietFont", full_path))
                        print(f"[FONT] Dùng font: {full_path}")
                        return "VietFont"
                    except Exception:
                        pass

    # Fallback: dùng Helvetica (không dấu đầy đủ nhưng vẫn tạo lớp invisible text)
    print("[FONT] Không tìm thấy font tiếng Việt, dùng Helvetica (lớp text vẫn searchable).")
    return "Helvetica"


def create_searchable_pdf(
    pages_data: list[dict],  # [{"img": np.ndarray, "blocks": [...], "page_size_pt": (w,h)}]
    output_path: str,
    font_name: str = "Helvetica",
):
    """
    Tạo PDF 2 lớp:
    - Lớp 1 (nền): Ảnh scan gốc
    - Lớp 2 (invisible text): Text từ OCR, đặt đúng vị trí bounding box
      → Người dùng có thể Ctrl+F, bôi đen, copy text
    """
    c = canvas.Canvas(output_path)

    for page_data in pages_data:
        img_array  = page_data["img"]           # numpy array (H, W, 3)
        blocks     = page_data["blocks"]
        w_pt, h_pt = page_data["page_size_pt"]  # kích thước trang PDF gốc (points)

        # Đặt kích thước trang theo PDF gốc
        c.setPageSize((w_pt, h_pt))

        # ── Lớp 1: Vẽ ảnh scan lên toàn trang ──
        pil_img = Image.fromarray(img_array)
        # ReportLab cần file tạm hoặc dùng drawInlineImage
        import io
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="JPEG", quality=85)
        img_buf.seek(0)

        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(img_buf), 0, 0, width=w_pt, height=h_pt)

        # ── Lớp 2: Invisible text (OCR result) ──
        # Tính tỉ lệ scale từ pixel ảnh → points PDF
        img_h, img_w = img_array.shape[:2]
        scale_x = w_pt / img_w
        scale_y = h_pt / img_h

        c.saveState()
        # Render mode 3 = invisible text (không hiển thị nhưng searchable)
        c.setFillAlpha(0)
        c.setFont(font_name, 1)  # font size tạm, sẽ tính lại từng block

        for block in blocks:
            text = block["text"]
            bbox = block["bbox"]

            # Chuyển tọa độ pixel → points
            x1_pt = bbox["x1"] * scale_x
            y1_pt = bbox["y1"] * scale_y
            x2_pt = bbox["x2"] * scale_x
            y2_pt = bbox["y2"] * scale_y

            box_w = x2_pt - x1_pt
            box_h = y2_pt - y1_pt

            if box_w <= 0 or box_h <= 0 or not text.strip():
                continue

            # PDF origin: bottom-left; ảnh origin: top-left → cần đảo Y
            pdf_y = h_pt - y2_pt  # bottom của bbox trong hệ PDF

            # Tính font size vừa với chiều cao bbox
            font_size = max(box_h * 0.85, 4)

            # Scale text vừa chiều ngang bbox bằng TextObject (hỗ trợ setHorizScale)
            text_width = c.stringWidth(text, font_name, font_size)
            h_scale = (box_w / text_width * 100) if text_width > 0 else 100

            # Dùng beginText để set render mode 3 (invisible) + horiz scale
            t = c.beginText(x1_pt, pdf_y)
            t.setFont(font_name, font_size)
            t.setTextRenderMode(3)   # invisible text, searchable
            t.setHorizScale(h_scale)
            t.textLine(text)
            c.drawText(t)

        c.restoreState()
        c.showPage()

    c.save()
    print(f"[PDF] Đã tạo Searchable PDF: {output_path}")


# ─────────────────────────────────────────────
# 5. LƯU JSON KẾT QUẢ OCR
# ─────────────────────────────────────────────
def save_json(pages_data: list[dict], output_path: str, source_pdf: str):
    output = {
        "document_id": Path(source_pdf).stem,
        "source_file": source_pdf,
        "total_pages": len(pages_data),
        "pages": []
    }

    for i, page_data in enumerate(pages_data):
        blocks = page_data["blocks"]
        img_h, img_w = page_data["img"].shape[:2]

        # Confidence tổng trang = trung bình các block
        confidences = [b["confidence"] for b in blocks]
        page_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Flag các block có confidence thấp (< 0.7) để UI highlight
        for b in blocks:
            b["low_confidence"] = b["confidence"] < 0.7

        output["pages"].append({
            "page_number": i + 1,
            "image_size": {"width": img_w, "height": img_h},
            "page_size_pt": {
                "width": page_data["page_size_pt"][0],
                "height": page_data["page_size_pt"][1]
            },
            "confidence_page": page_conf,
            "text_blocks": blocks,
            "full_text": " ".join(b["text"] for b in blocks),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[JSON] Đã lưu kết quả OCR: {output_path}")


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    input_pdf: str,
    output_dir: str,
    dpi: int = 200,
    device: str = "gpu",
    font_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_pdf).stem
    out_pdf  = os.path.join(output_dir, f"{stem}_searchable.pdf")
    out_json = os.path.join(output_dir, f"{stem}_ocr.json")

    t_start = time.time()

    # Bước 1: Convert PDF → images
    pages_imgs = pdf_to_images(input_pdf, dpi=dpi)

    # Bước 2: Khởi tạo OCR
    ocr = init_ocr(device=device)

    # Bước 3: OCR từng trang
    pages_data = []
    print(f"\n[OCR] Đang nhận dạng {len(pages_imgs)} trang...\n")
    for i, (img, page_size_pt) in enumerate(tqdm(pages_imgs, desc="Trang")):
        blocks = ocr_page(ocr, img)
        pages_data.append({
            "img": img,
            "blocks": blocks,
            "page_size_pt": page_size_pt,
        })

        # In preview trang đầu
        if i == 0:
            print(f"\n  ── Preview trang 1: {len(blocks)} đoạn text ──")
            for b in blocks[:5]:
                flag = " ⚠️" if b["confidence"] < 0.7 else ""
                print(f"    [{b['confidence']:.0%}]{flag} {b['text'][:60]}")
            if len(blocks) > 5:
                print(f"    ... và {len(blocks)-5} đoạn nữa\n")

    # Bước 4: Đăng ký font & tạo Searchable PDF
    font_name = register_vietnamese_font(font_dir)
    create_searchable_pdf(pages_data, out_pdf, font_name)

    # Bước 5: Lưu JSON
    save_json(pages_data, out_json, input_pdf)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"✅ Hoàn thành! Thời gian: {elapsed:.1f}s ({elapsed/len(pages_imgs):.1f}s/trang)")
    print(f"   📄 Searchable PDF : {out_pdf}")
    print(f"   📋 JSON OCR       : {out_json}")
    print(f"{'='*50}\n")

    return out_pdf, out_json


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Pipeline: Scanned PDF → Searchable PDF + JSON")
    parser.add_argument("--input",    required=True,  help="Đường dẫn file PDF scan đầu vào")
    parser.add_argument("--output",   default="output", help="Thư mục lưu kết quả (mặc định: output/)")
    parser.add_argument("--dpi",      type=int, default=200, help="DPI render ảnh (mặc định: 200)")
    parser.add_argument("--device",   default="gpu",  choices=["gpu", "cpu"], help="Thiết bị chạy OCR")
    parser.add_argument("--font_dir", default=None,   help="Thư mục chứa font TTF (tuỳ chọn)")
    args = parser.parse_args()

    run_pipeline(
        input_pdf  = args.input,
        output_dir = args.output,
        dpi        = args.dpi,
        device     = args.device,
        font_dir   = args.font_dir,
    )
