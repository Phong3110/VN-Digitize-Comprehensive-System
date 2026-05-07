"""
OCR Pipeline v4 — FINAL
Scanned PDF → Searchable PDF/A + JSON

Architecture:
  1. PaddleOCR  → Detection (PP-OCRv5_server_det) — tìm vị trí text block
  2. split_box_to_lines() → Tách block nhiều dòng thành từng dòng đơn
  3. VietOCR (vgg_transformer) → Recognition — đọc chữ tiếng Việt

Tối ưu VRAM 4GB (RTX 3050 Laptop):
  - Chạy Detection xong → giải phóng VRAM Paddle → mới load VietOCR
  - VietOCR chạy batch_size=16

Yêu cầu:
    pip install pymupdf paddleocr reportlab tqdm
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install vietocr gdown==4.6.0

Sử dụng:
    python ocr_pipeline_v4.py --input scan.pdf --output output --dpi 200
    python ocr_pipeline_v4.py --input scan.pdf --output output --skip_ocr
    python ocr_pipeline_v4.py --input scan.pdf --output output --beamsearch  (chậm hơn, chuẩn hơn)
"""

import argparse
import io
import json
import os
import time
import unicodedata
from pathlib import Path

import fitz
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

MAX_SIDE = 3500  # chia tile nếu ảnh lớn hơn (pixel)
PADDING  = 3     # padding quanh mỗi crop (pixel)


# ══════════════════════════════════════════════
# 1. KHỞI TẠO MODELS
# ══════════════════════════════════════════════
def init_paddle_detector(device: str = "gpu") -> PaddleOCR:
    """Chỉ load Detection — không load Recognition để tiết kiệm VRAM."""
    print("[DET] Khởi tạo PaddleOCR detector (PP-OCRv5_server_det)...")
    ocr = PaddleOCR(
        # Pre-processing modules
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,

        # Chỉ định thẳng detection model
        text_detection_model_name="PP-OCRv5_server_det",

        device=device,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=2.0,
    )
    print("[DET] Sẵn sàng.\n")
    return ocr


def init_vietocr(device: str = "cuda:0", beamsearch: bool = False) -> Predictor:
    """Load VietOCR vgg_transformer."""
    print(f"[REC] Khởi tạo VietOCR (beamsearch={beamsearch})...")
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device
    config["predictor"]["beamsearch"] = beamsearch
    predictor = Predictor(config)
    print("[REC] Sẵn sàng.\n")
    return predictor


# ══════════════════════════════════════════════
# 2. CONVERT PDF → IMAGES
# ══════════════════════════════════════════════
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    """Trả về list (numpy_RGB, (w_pt, h_pt))."""
    doc  = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pages = []

    print(f"[PDF] Đọc {len(doc)} trang từ '{pdf_path}' (DPI={dpi})...")
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        pages.append((img, (page.rect.width, page.rect.height)))

    doc.close()
    return pages


# ══════════════════════════════════════════════
# 3. DETECTION
# ══════════════════════════════════════════════
def detect_page(ocr: PaddleOCR, img: np.ndarray) -> list:
    """
    Detect text blocks trên 1 trang.
    Tự động chia tile nếu ảnh > MAX_SIDE px (tránh PaddleOCR tự resize).
    Trả về list polygon boxes.
    """
    h, w = img.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return _detect_single(ocr, img, offset_y=0)

    # Chia tile dọc với overlap
    overlap   = 150
    boxes_all = []
    y = 0
    while y < h:
        y_end      = min(y + MAX_SIDE, h)
        tile       = img[y:y_end, :]
        tile_boxes = _detect_single(ocr, tile, offset_y=y)

        # Bỏ phần overlap với tile trước để tránh detect trùng
        if boxes_all and y > 0:
            tile_boxes = [b for b in tile_boxes if b["y_min"] >= y + overlap]

        boxes_all.extend(tile_boxes)
        if y_end == h:
            break
        y = y_end - overlap

    return boxes_all


def _detect_single(ocr: PaddleOCR, img: np.ndarray, offset_y: int) -> list:
    results = ocr.predict(img)
    boxes   = []
    for res in results:
        if res is None:
            continue
        for poly in res.get("rec_polys", []):
            polygon = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            polygon = [[p[0], p[1] + offset_y] for p in polygon]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            boxes.append({
                "polygon": polygon,
                "x_min": int(min(xs)), "y_min": int(min(ys)),
                "x_max": int(max(xs)), "y_max": int(max(ys)),
            })
    return boxes


# ══════════════════════════════════════════════
# 4. TÁCH BLOCK NHIỀU DÒNG → DÒNG ĐƠN
# ══════════════════════════════════════════════
def split_box_to_lines(box: dict, img_w: int) -> list:
    """
    PaddleOCR đôi khi group cả đoạn văn nhiều dòng thành 1 box.
    VietOCR chỉ đọc được 1 dòng đơn → cần tách ra.

    Ước tính số dòng dựa trên chiều cao box vs chiều cao 1 dòng ước tính.
    Chiều cao 1 dòng ~ 3% chiều rộng trang A4 (thực nghiệm).
    """
    x1, y1 = box["x_min"], box["y_min"]
    x2, y2 = box["x_max"], box["y_max"]
    box_h  = y2 - y1

    # Ước tính chiều cao 1 dòng
    estimated_line_h = max(20, int(img_w * 0.030))
    n_lines = max(1, round(box_h / estimated_line_h))

    if n_lines == 1:
        return [(x1, y1, x2, y2)]

    # Chia đều
    line_h = box_h // n_lines
    lines  = []
    for i in range(n_lines):
        ly1 = y1 + i * line_h
        ly2 = y1 + (i + 1) * line_h if i < n_lines - 1 else y2
        lines.append((x1, ly1, x2, ly2))

    return lines


# ══════════════════════════════════════════════
# 5. RECOGNITION — VietOCR
# ══════════════════════════════════════════════
def recognize_page(
    predictor: Predictor,
    img:       np.ndarray,
    boxes:     list,
    batch_size: int = 16,
) -> list:
    """
    Với mỗi box:
      1. Tách thành dòng đơn (split_box_to_lines)
      2. Crop + padding
      3. Batch qua VietOCR
    Trả về list text blocks với bbox từng dòng.
    """
    if not boxes:
        return []

    img_h, img_w = img.shape[:2]

    # Chuẩn bị tất cả line crops
    line_crops = []   # PIL Images
    line_bboxes = []  # (x1, y1, x2, y2) tương ứng

    for box in boxes:
        lines = split_box_to_lines(box, img_w)
        for (x1, y1, x2, y2) in lines:
            # Padding + clamp
            x1c = max(0, x1 - PADDING)
            y1c = max(0, y1 - PADDING)
            x2c = min(img_w, x2 + PADDING)
            y2c = min(img_h, y2 + PADDING)

            if x2c <= x1c or y2c <= y1c:
                continue

            crop = Image.fromarray(img[y1c:y2c, x1c:x2c])
            line_crops.append(crop)
            line_bboxes.append((x1, y1, x2, y2))

    if not line_crops:
        return []

    # Chạy VietOCR theo batch
    texts = []
    for i in range(0, len(line_crops), batch_size):
        batch      = line_crops[i:i + batch_size]
        batch_text = predictor.predict_batch(batch)
        texts.extend(batch_text)

    # Ghép kết quả
    blocks = []
    for (x1, y1, x2, y2), text in zip(line_bboxes, texts):
        text = unicodedata.normalize("NFC", text.strip())
        if not text or not is_valid_text(text):
            continue

        # VietOCR không có confidence score
        # Heuristic: text ngắn (<3 ký tự) thường là nhận sai
        confidence = 0.95 if len(text) >= 3 else 0.60

        blocks.append({
            "text":           text,
            "confidence":     confidence,
            "bbox":           {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "low_confidence": confidence < 0.7,
        })

    return blocks


def is_valid_text(text: str) -> bool:
    """
    Lọc bỏ các text vô nghĩa:
    - Toàn số dài (QR code, barcode, số hiệu)
    - Quá ngắn (1-2 ký tự)
    - Không chứa chữ cái nào
    """
    import re
    text = text.strip()

    # Quá ngắn
    if len(text) < 2:
        return False

    # Toàn số (barcode/QR) — dài hơn 6 ký tự
    if re.fullmatch(r'[\d\s\-\.]+', text) and len(text) > 6:
        return False

    # Không có chữ cái nào (chỉ có ký hiệu, số)
    if not re.search(r'[a-zA-ZÀ-ỹ]', text):
        return False

    return True

# ══════════════════════════════════════════════
# 6. TẠO SEARCHABLE PDF (2 lớp)
# ══════════════════════════════════════════════
def register_font(font_dir: str = None) -> str:
    dirs = ([font_dir] if font_dir else []) + [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]
    candidates = {
        "Arial":   ["arial.ttf",   "Arial.ttf"],
        "Tahoma":  ["tahoma.ttf",  "Tahoma.ttf"],
        "Calibri": ["calibri.ttf", "Calibri.ttf"],
    }
    for name, files in candidates.items():
        for fname in files:
            for d in dirs:
                if not d:
                    continue
                full = os.path.join(d, fname)
                if os.path.exists(full):
                    try:
                        pdfmetrics.registerFont(TTFont("VietFont", full))
                        print(f"[FONT] {full}")
                        return "VietFont"
                    except Exception:
                        pass
    print("[FONT] Dùng Helvetica (fallback).")
    return "Helvetica"


def create_searchable_pdf(pages_data: list, output_path: str, font_name: str):
    """
    PDF 2 lớp:
    - Lớp 1 (nền): ảnh scan gốc
    - Lớp 2 (invisible text): text VietOCR đặt đúng vị trí → Ctrl+F / bôi đen được
    """
    c = canvas.Canvas(output_path)

    for page_data in tqdm(pages_data, desc="Tạo PDF"):
        img_array  = page_data["img"]
        blocks     = page_data["blocks"]
        w_pt, h_pt = page_data["page_size_pt"]

        c.setPageSize((w_pt, h_pt))

        # Lớp 1: ảnh scan
        buf = io.BytesIO()
        Image.fromarray(img_array).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=w_pt, height=h_pt)

        # Lớp 2: invisible text
        img_h, img_w = img_array.shape[:2]
        sx = w_pt / img_w
        sy = h_pt / img_h

        for block in blocks:
            text = block["text"]
            bbox = block["bbox"]

            x1_pt = bbox["x1"] * sx
            y2_pt = bbox["y2"] * sy
            box_w = (bbox["x2"] - bbox["x1"]) * sx
            box_h = (bbox["y2"] - bbox["y1"]) * sy

            if box_w <= 0 or box_h <= 0 or not text:
                continue

            pdf_y     = h_pt - y2_pt          # đảo Y (PDF: bottom-left)
            font_size = max(box_h * 0.85, 4)
            text_w    = c.stringWidth(text, font_name, font_size)
            h_scale   = (box_w / text_w * 100) if text_w > 0 else 100

            t = c.beginText(x1_pt, pdf_y)
            t.setFont(font_name, font_size)
            t.setTextRenderMode(3)             # invisible, searchable
            t.setHorizScale(h_scale)
            t.textLine(text)
            c.drawText(t)

        c.showPage()

    c.save()
    print(f"[PDF] Lưu: {output_path}")


# ══════════════════════════════════════════════
# 7. JSON CACHE
# ══════════════════════════════════════════════
def save_json(pages_data: list, output_path: str, source_pdf: str):
    out = {
        "document_id": Path(source_pdf).stem,
        "source_file": source_pdf,
        "total_pages": len(pages_data),
        "pages":       [],
    }
    for i, pd in enumerate(pages_data):
        img_h, img_w = pd["img"].shape[:2]
        blocks = pd["blocks"]
        confs  = [b["confidence"] for b in blocks]
        out["pages"].append({
            "page_number":     i + 1,
            "image_size":      {"width": img_w, "height": img_h},
            "page_size_pt":    {"width": pd["page_size_pt"][0], "height": pd["page_size_pt"][1]},
            "confidence_page": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "text_blocks":     blocks,
            "full_text":       " ".join(b["text"] for b in blocks),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[JSON] Lưu: {output_path}")


def load_json_cache(json_path: str, pages_imgs: list) -> list:
    print(f"[CACHE] Đọc từ '{json_path}'...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "img":          img,
            "blocks":       data["pages"][i]["text_blocks"],
            "page_size_pt": page_size_pt,
        }
        for i, (img, page_size_pt) in enumerate(pages_imgs)
    ]


# ══════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════
def run_pipeline(
    input_pdf:  str,
    output_dir: str,
    dpi:        int  = 200,
    device:     str  = "gpu",
    font_dir:   str  = None,
    skip_ocr:   bool = False,
    beamsearch: bool = False,
    batch_size: int  = 16,
):
    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_pdf).stem
    out_pdf  = os.path.join(output_dir, f"{stem}_searchable.pdf")
    out_json = os.path.join(output_dir, f"{stem}_ocr.json")
    t_start  = time.time()

    # Bước 1: PDF → images
    pages_imgs = pdf_to_images(input_pdf, dpi=dpi)

    if skip_ocr and os.path.exists(out_json):
        # Dùng cache, bỏ qua OCR
        pages_data = load_json_cache(out_json, pages_imgs)

    else:
        paddle_device = "gpu" if device in ("gpu", "cuda") else "cpu"
        torch_device  = "cuda:0" if device in ("gpu", "cuda") else "cpu"

        # ── Bước 2: DETECTION (PaddleOCR) ──
        detector = init_paddle_detector(paddle_device)
        print(f"[DET] Đang detect {len(pages_imgs)} trang...\n")

        all_boxes = []
        for img, _ in tqdm(pages_imgs, desc="Detection"):
            all_boxes.append(detect_page(detector, img))

        # Giải phóng VRAM Paddle trước khi load VietOCR
        del detector
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        torch.cuda.empty_cache()
        print(f"\n[DET] Xong. Đã giải phóng VRAM Paddle.\n")

        # ── Bước 3: RECOGNITION (VietOCR) ──
        predictor = init_vietocr(torch_device, beamsearch=beamsearch)
        print(f"[REC] Đang nhận dạng {len(pages_imgs)} trang...\n")

        pages_data = []
        for i, ((img, page_size_pt), boxes) in enumerate(
            tqdm(zip(pages_imgs, all_boxes), total=len(pages_imgs), desc="Recognition")
        ):
            blocks = recognize_page(predictor, img, boxes, batch_size=batch_size)
            pages_data.append({
                "img":          img,
                "blocks":       blocks,
                "page_size_pt": page_size_pt,
            })

            # Preview trang đầu
            if i == 0:
                print(f"\n  ── Preview trang 1: {len(blocks)} dòng ──")
                for b in blocks[:10]:
                    print(f"    {b['text'][:80]}")
                if len(blocks) > 10:
                    print(f"    ... và {len(blocks)-10} dòng nữa\n")

        # Lưu JSON cache TRƯỚC khi tạo PDF — tránh mất kết quả nếu PDF lỗi
        save_json(pages_data, out_json, input_pdf)

    # ── Bước 4: Tạo Searchable PDF ──
    font_name = register_font(font_dir)
    create_searchable_pdf(pages_data, out_pdf, font_name)

    elapsed = time.time() - t_start
    n = len(pages_imgs)
    print(f"\n{'═'*55}")
    print(f"✅ Hoàn thành! {elapsed:.1f}s tổng ({elapsed/n:.1f}s/trang)")
    print(f"   📄 Searchable PDF : {out_pdf}")
    print(f"   📋 JSON OCR       : {out_json}")
    print(f"{'═'*55}\n")

    return out_pdf, out_json


# ══════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Pipeline v4: PaddleOCR det + split lines + VietOCR rec"
    )
    parser.add_argument("--input",      required=True,         help="File PDF scan đầu vào")
    parser.add_argument("--output",     default="output",      help="Thư mục lưu kết quả")
    parser.add_argument("--dpi",        type=int, default=200, help="DPI render (mặc định 200)")
    parser.add_argument("--device",     default="gpu",         choices=["gpu", "cpu"])
    parser.add_argument("--font_dir",   default=None,          help="Thư mục font TTF tùy chọn")
    parser.add_argument("--skip_ocr",   action="store_true",   help="Dùng JSON cache, bỏ qua OCR")
    parser.add_argument("--beamsearch", action="store_true",   help="Bật beamsearch (chuẩn hơn, chậm ~3x)")
    parser.add_argument("--batch_size", type=int, default=16,  help="Batch size VietOCR (mặc định 16)")
    args = parser.parse_args()

    run_pipeline(
        input_pdf  = args.input,
        output_dir = args.output,
        dpi        = args.dpi,
        device     = args.device,
        font_dir   = args.font_dir,
        skip_ocr   = args.skip_ocr,
        beamsearch = args.beamsearch,
        batch_size = args.batch_size,
    )