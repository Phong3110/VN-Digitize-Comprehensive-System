"""
OCR Pipeline: Scanned PDF → Searchable PDF/A + JSON
Dự án: VN-Digitize - Module 1 & 2

Cải tiến v2:
  - lang="vi" (Latin PP-OCRv5 mobile) — nhận dạng dấu tiếng Việt tốt hơn
  - Tự động chia ảnh lớn thành tiles để tránh resize mất chất lượng
  - Unicode NFC normalization — ghép dấu rời thành ký tự hoàn chỉnh
  - Cache JSON trung gian — không phải OCR lại khi lỗi ở bước tạo PDF

Yêu cầu cài đặt:
    pip install pymupdf paddleocr reportlab tqdm

Sử dụng:
    python ocr_pipeline.py --input scan.pdf --output output_folder --dpi 200
    python ocr_pipeline.py --input scan.pdf --output output_folder --skip_ocr
"""

import argparse
import io
import json
import os
import time
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tqdm import tqdm

# Giới hạn cạnh dài tối đa trước khi chia tile (pixel)
MAX_SIDE = 3500


# ─────────────────────────────────────────────
# 1. KHỞI TẠO PADDLEOCR
# ─────────────────────────────────────────────
def init_ocr(device: str = "gpu") -> PaddleOCR:
    print(f"[OCR] Khởi tạo PaddleOCR trên {device.upper()}...")
    ocr = PaddleOCR(
        # Module 1: Image Pre-processing
        use_doc_orientation_classify=True,  # Tự động xoay tài liệu 0/90/180/270°
        use_doc_unwarping=True,             # Nắn thẳng ảnh bị cong (scan sách)
        use_textline_orientation=True,       # Phát hiện dòng chữ bị lật ngược

        # Module 2: OCR Engine
        # lang="vi" → latin_PP-OCRv5_mobile_rec
        # Đã confirm nhận dạng dấu tiếng Việt tốt hơn server_rec (train chủ yếu CJK)
        lang="vi",
        ocr_version="PP-OCRv5",
        device=device,

        # Tham số detect — limit_type="max" giới hạn cạnh DÀI (phù hợp ảnh A4 dọc)
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=2.0,
    )
    print("[OCR] Sẵn sàng.\n")
    return ocr


# ─────────────────────────────────────────────
# 2. UNICODE NORMALIZATION
# ─────────────────────────────────────────────
def normalize_viet(text: str) -> str:
    """
    Chuẩn hóa Unicode NFC: ghép ký tự + dấu rời → 1 ký tự hoàn chỉnh.
    VD: "ă" (2 codepoints) → "ă" (1 codepoint)
    """
    return unicodedata.normalize("NFC", text)


# ─────────────────────────────────────────────
# 3. CONVERT PDF → IMAGES
# ─────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """Trả về list (numpy_image_RGB, (width_pt, height_pt))"""
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    print(f"[PDF] Đang đọc {len(doc)} trang từ '{pdf_path}' (DPI={dpi})...")
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        page_size_pt = (page.rect.width, page.rect.height)
        pages.append((img, page_size_pt))

    doc.close()
    return pages


# ─────────────────────────────────────────────
# 4. TILE-BASED OCR — tránh resize ảnh lớn
# ─────────────────────────────────────────────
def ocr_image_tiled(ocr: PaddleOCR, img: np.ndarray, max_side: int = MAX_SIDE) -> list:
    """
    Nếu ảnh có cạnh > max_side: chia tile dọc (overlap 150px) → tránh bị
    PaddleOCR tự resize xuống 4000px làm mất chi tiết chữ nhỏ.
    """
    h, w = img.shape[:2]

    if max(h, w) <= max_side:
        return _ocr_single(ocr, img, offset_x=0, offset_y=0)

    overlap = 150
    tile_h = max_side
    blocks_all = []
    y = 0

    while y < h:
        y_end = min(y + tile_h, h)
        tile = img[y:y_end, 0:w]
        tile_blocks = _ocr_single(ocr, tile, offset_x=0, offset_y=y)

        # Bỏ qua các block trong vùng overlap của tile trước (tránh trùng lặp)
        if blocks_all and y > 0:
            tile_blocks = [b for b in tile_blocks if b["bbox"]["y1"] >= y + overlap]

        blocks_all.extend(tile_blocks)

        if y_end == h:
            break
        y = y_end - overlap

    return blocks_all


def _ocr_single(ocr: PaddleOCR, img: np.ndarray, offset_x: int, offset_y: int) -> list:
    """OCR 1 ảnh, trả về blocks với tọa độ đã cộng offset."""
    results = ocr.predict(img)
    blocks = []

    for res in results:
        if res is None:
            continue

        polys  = res.get("rec_polys", [])
        texts  = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            text = normalize_viet(text.strip())
            if not text:
                continue

            polygon = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            polygon = [[p[0] + offset_x, p[1] + offset_y] for p in polygon]

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            blocks.append({
                "text": text,
                "confidence": round(float(score), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "polygon": polygon,
            })

    return blocks


# ─────────────────────────────────────────────
# 5. TẠO SEARCHABLE PDF (2 lớp)
# ─────────────────────────────────────────────
def register_vietnamese_font(font_dir: str = None) -> str:
    search_dirs = []
    if font_dir:
        search_dirs.append(font_dir)
    search_dirs += [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]

    candidates = {
        "Arial":   ["arial.ttf", "Arial.ttf"],
        "Tahoma":  ["tahoma.ttf", "Tahoma.ttf"],
        "Calibri": ["calibri.ttf", "Calibri.ttf"],
        "Times":   ["times.ttf", "Times.ttf"],
    }

    for name, files in candidates.items():
        for fname in files:
            for d in search_dirs:
                full = os.path.join(d, fname)
                if os.path.exists(full):
                    try:
                        pdfmetrics.registerFont(TTFont("VietFont", full))
                        print(f"[FONT] Dùng font: {full}")
                        return "VietFont"
                    except Exception:
                        pass

    print("[FONT] Dùng Helvetica (fallback).")
    return "Helvetica"


def create_searchable_pdf(pages_data: list, output_path: str, font_name: str = "Helvetica"):
    """
    PDF 2 lớp:
    - Lớp 1: ảnh scan gốc
    - Lớp 2: invisible text đúng vị trí → Ctrl+F / bôi đen / copy được
    """
    c = canvas.Canvas(output_path)

    for page_data in tqdm(pages_data, desc="Tạo PDF"):
        img_array  = page_data["img"]
        blocks     = page_data["blocks"]
        w_pt, h_pt = page_data["page_size_pt"]

        c.setPageSize((w_pt, h_pt))

        # Lớp 1: ảnh scan
        pil_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=w_pt, height=h_pt)

        # Lớp 2: invisible text
        img_h, img_w = img_array.shape[:2]
        scale_x = w_pt / img_w
        scale_y = h_pt / img_h

        for block in blocks:
            text = block["text"]
            bbox = block["bbox"]

            x1_pt = bbox["x1"] * scale_x
            y1_pt = bbox["y1"] * scale_y
            x2_pt = bbox["x2"] * scale_x
            y2_pt = bbox["y2"] * scale_y
            box_w = x2_pt - x1_pt
            box_h = y2_pt - y1_pt

            if box_w <= 0 or box_h <= 0 or not text:
                continue

            pdf_y      = h_pt - y2_pt          # đảo Y: PDF bottom-left, ảnh top-left
            font_size  = max(box_h * 0.85, 4)
            text_w     = c.stringWidth(text, font_name, font_size)
            h_scale    = (box_w / text_w * 100) if text_w > 0 else 100

            t = c.beginText(x1_pt, pdf_y)
            t.setFont(font_name, font_size)
            t.setTextRenderMode(3)          # invisible, searchable
            t.setHorizScale(h_scale)
            t.textLine(text)
            c.drawText(t)

        c.showPage()

    c.save()
    print(f"[PDF] Searchable PDF đã lưu: {output_path}")


# ─────────────────────────────────────────────
# 6. LƯU / ĐỌC JSON CACHE
# ─────────────────────────────────────────────
def save_json(pages_data: list, output_path: str, source_pdf: str):
    output = {
        "document_id": Path(source_pdf).stem,
        "source_file": source_pdf,
        "total_pages": len(pages_data),
        "pages": [],
    }

    for i, pd in enumerate(pages_data):
        blocks   = pd["blocks"]
        img_h, img_w = pd["img"].shape[:2]
        confs    = [b["confidence"] for b in blocks]
        page_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        for b in blocks:
            b["low_confidence"] = b["confidence"] < 0.7

        output["pages"].append({
            "page_number":   i + 1,
            "image_size":    {"width": img_w, "height": img_h},
            "page_size_pt":  {"width": pd["page_size_pt"][0], "height": pd["page_size_pt"][1]},
            "confidence_page": page_conf,
            "text_blocks":   blocks,
            "full_text":     " ".join(b["text"] for b in blocks),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[JSON] Đã lưu: {output_path}")


def load_json_cache(json_path: str, pages_imgs: list) -> list:
    """Đọc JSON cache và ghép lại với ảnh để tạo PDF không cần OCR lại."""
    print(f"[CACHE] Đọc từ '{json_path}'...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pages_data = []
    for i, (img, page_size_pt) in enumerate(pages_imgs):
        page_json = data["pages"][i]
        pages_data.append({
            "img":          img,
            "blocks":       page_json["text_blocks"],
            "page_size_pt": page_size_pt,
        })
    return pages_data


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    input_pdf:  str,
    output_dir: str,
    dpi:        int  = 200,
    device:     str  = "gpu",
    font_dir:   str  = None,
    skip_ocr:   bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_pdf).stem
    out_pdf  = os.path.join(output_dir, f"{stem}_searchable.pdf")
    out_json = os.path.join(output_dir, f"{stem}_ocr.json")

    t_start = time.time()

    # Bước 1: Convert PDF → images (luôn cần để embed vào PDF)
    pages_imgs = pdf_to_images(input_pdf, dpi=dpi)

    if skip_ocr and os.path.exists(out_json):
        # Chế độ skip: dùng JSON cache, bỏ qua OCR
        pages_data = load_json_cache(out_json, pages_imgs)
    else:
        # Chế độ đầy đủ: chạy OCR
        ocr = init_ocr(device=device)
        pages_data = []
        print(f"\n[OCR] Đang nhận dạng {len(pages_imgs)} trang...\n")

        for i, (img, page_size_pt) in enumerate(tqdm(pages_imgs, desc="OCR trang")):
            blocks = ocr_image_tiled(ocr, img)
            pages_data.append({
                "img":          img,
                "blocks":       blocks,
                "page_size_pt": page_size_pt,
            })

            if i == 0:
                print(f"\n  ── Preview trang 1: {len(blocks)} đoạn ──")
                for b in blocks[:8]:
                    flag = " ⚠️" if b["confidence"] < 0.7 else ""
                    print(f"    [{b['confidence']:.0%}]{flag} {b['text'][:70]}")
                if len(blocks) > 8:
                    print(f"    ... và {len(blocks)-8} đoạn nữa\n")

        # Lưu JSON cache TRƯỚC khi tạo PDF — tránh mất kết quả OCR nếu PDF lỗi
        save_json(pages_data, out_json, input_pdf)

    # Bước cuối: tạo Searchable PDF
    font_name = register_vietnamese_font(font_dir)
    create_searchable_pdf(pages_data, out_pdf, font_name)

    elapsed = time.time() - t_start
    n = len(pages_imgs)
    print(f"\n{'='*55}")
    print(f"✅ Hoàn thành! {elapsed:.1f}s tổng ({elapsed/n:.1f}s/trang)")
    print(f"   📄 Searchable PDF : {out_pdf}")
    print(f"   📋 JSON OCR       : {out_json}")
    print(f"{'='*55}\n")

    return out_pdf, out_json


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Pipeline v2: Scanned PDF → Searchable PDF + JSON"
    )
    parser.add_argument("--input",    required=True,        help="File PDF scan đầu vào")
    parser.add_argument("--output",   default="output",     help="Thư mục lưu kết quả")
    parser.add_argument("--dpi",      type=int, default=200, help="DPI render (mặc định 200)")
    parser.add_argument("--device",   default="gpu",        choices=["gpu", "cpu"])
    parser.add_argument("--font_dir", default=None,         help="Thư mục font TTF tùy chọn")
    parser.add_argument("--skip_ocr", action="store_true",
                        help="Bỏ qua OCR, dùng JSON cache có sẵn để tạo lại PDF")
    args = parser.parse_args()

    run_pipeline(
        input_pdf  = args.input,
        output_dir = args.output,
        dpi        = args.dpi,
        device     = args.device,
        font_dir   = args.font_dir,
        skip_ocr   = args.skip_ocr,
    )
