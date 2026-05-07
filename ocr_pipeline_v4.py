"""
OCR Pipeline v3: Scanned PDF → Searchable PDF/A + JSON
Dự án: VN-Digitize - Module 1 & 2

Architecture:
  - PaddleOCR  → Detection (tìm vị trí text) + 3 module pre-processing
  - VietOCR    → Recognition (đọc chữ tiếng Việt)

Tối ưu VRAM 4GB (RTX 3050 Laptop):
  - Load PaddleOCR det trước, release cache sau detect
  - Load VietOCR rec sau, chạy batch nhỏ
  - Không load cả 2 full model cùng lúc

Yêu cầu:
    pip install pymupdf paddleocr vietocr reportlab tqdm gdown==4.6.0
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Sử dụng:
    python ocr_pipeline_v3.py --input scan.pdf --output output --dpi 200
    python ocr_pipeline_v3.py --input scan.pdf --output output --skip_ocr
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

MAX_SIDE = 3500  # pixel — chia tile nếu ảnh lớn hơn


# ─────────────────────────────────────────────
# 1. KHỞI TẠO MODELS
# ─────────────────────────────────────────────
def init_paddle_detector(device: str = "gpu") -> PaddleOCR:
    """
    Chỉ load phần DETECTION của PaddleOCR + 3 module pre-processing.
    Tắt recognition để tiết kiệm VRAM.
    """
    print("[DET] Khởi tạo PaddleOCR detector...")
    ocr = PaddleOCR(
        # Module 1: Pre-processing
        use_doc_orientation_classify=True,   # Xoay tài liệu 0/90/180/270°
        use_doc_unwarping=False,              # Nắn thẳng ảnh cong True
        use_textline_orientation=True,        # Phát hiện dòng bị lật

        # Chỉ detect, không rec
        text_detection_model_name="PP-OCRv5_server_det",
        ocr_version="PP-OCRv5",
        device=device,

        # Tham số detect
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=2.0,
    )
    print("[DET] Sẵn sàng.\n")
    return ocr


def init_vietocr(device: str = "cuda:0", beamsearch: bool = False) -> Predictor:
    """
    Load VietOCR vgg_transformer.
    beamsearch=True: chính xác hơn nhưng chậm ~3x
    beamsearch=False: nhanh, đủ dùng cho production
    """
    print("[REC] Khởi tạo VietOCR...")
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device
    config["predictor"]["beamsearch"] = beamsearch
    predictor = Predictor(config)
    print("[REC] Sẵn sàng.\n")
    return predictor


# ─────────────────────────────────────────────
# 2. CONVERT PDF → IMAGES
# ─────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    print(f"[PDF] Đang đọc {len(doc)} trang (DPI={dpi})...")
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        pages.append((img, (page.rect.width, page.rect.height)))

    doc.close()
    return pages


# ─────────────────────────────────────────────
# 3. DETECTION — tìm bounding boxes
# ─────────────────────────────────────────────
def detect_page(ocr: PaddleOCR, img: np.ndarray, max_side: int = MAX_SIDE) -> list:
    """
    Chạy detection, trả về list polygon boxes.
    Tự động chia tile nếu ảnh > max_side px.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return _detect_single(ocr, img, offset_y=0)

    # Chia tile dọc
    overlap = 150
    boxes_all = []
    y = 0
    while y < h:
        y_end = min(y + max_side, h)
        tile = img[y:y_end, :]
        tile_boxes = _detect_single(ocr, tile, offset_y=y)

        if boxes_all and y > 0:
            tile_boxes = [b for b in tile_boxes if b["y_min"] >= y + overlap]

        boxes_all.extend(tile_boxes)
        if y_end == h:
            break
        y = y_end - overlap

    return boxes_all


def _detect_single(ocr: PaddleOCR, img: np.ndarray, offset_y: int) -> list:
    results = ocr.predict(img)
    boxes = []
    for res in results:
        if res is None:
            continue
        for poly in res.get("rec_polys", []):
            polygon = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            # Cộng offset Y cho tile
            polygon = [[p[0], p[1] + offset_y] for p in polygon]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            boxes.append({
                "polygon": polygon,
                "x_min": int(min(xs)), "y_min": int(min(ys)),
                "x_max": int(max(xs)), "y_max": int(max(ys)),
            })
    return boxes


# ─────────────────────────────────────────────
# 4. CROP TEXT LINE — chuẩn bị input cho VietOCR
# ─────────────────────────────────────────────
def crop_text_region(img: np.ndarray, box: dict, padding: int = 4) -> Image.Image:
    """
    Crop vùng chữ theo bounding box, thêm padding nhỏ.
    Trả về PIL Image để đưa vào VietOCR.
    """
    h, w = img.shape[:2]
    x1 = max(0, box["x_min"] - padding)
    y1 = max(0, box["y_min"] - padding)
    x2 = min(w, box["x_max"] + padding)
    y2 = min(h, box["y_max"] + padding)

    crop = img[y1:y2, x1:x2]
    return Image.fromarray(crop)


# ─────────────────────────────────────────────
# 5. RECOGNITION — VietOCR đọc chữ từng crop
# ─────────────────────────────────────────────
def recognize_page(
    predictor: Predictor,
    img: np.ndarray,
    boxes: list,
    batch_size: int = 16,
) -> list:
    """
    Crop từng bbox → batch đưa vào VietOCR → trả về list text blocks.
    batch_size=16 phù hợp VRAM 4GB.
    """
    if not boxes:
        return []

    # Crop tất cả regions
    crops = [crop_text_region(img, box) for box in boxes]

    # Chạy VietOCR theo batch
    texts = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        batch_texts = predictor.predict_batch(batch)
        texts.extend(batch_texts)

    # Ghép kết quả
    blocks = []
    for box, text in zip(boxes, texts):
        text = unicodedata.normalize("NFC", text.strip())
        if not text:
            continue

        # VietOCR không trả confidence → ước tính dựa trên độ dài text
        # (block rỗng hoặc 1 ký tự thường là nhận sai)
        confidence = 0.95 if len(text) > 2 else 0.60

        blocks.append({
            "text": text,
            "confidence": confidence,
            "bbox": {
                "x1": box["x_min"], "y1": box["y_min"],
                "x2": box["x_max"], "y2": box["y_max"],
            },
            "polygon": box["polygon"],
            "low_confidence": confidence < 0.7,
        })

    return blocks


# ─────────────────────────────────────────────
# 6. TẠO SEARCHABLE PDF (2 lớp)
# ─────────────────────────────────────────────
def register_font(font_dir: str = None) -> str:
    search_dirs = [font_dir] if font_dir else []
    search_dirs += [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]
    candidates = {
        "Arial":   ["arial.ttf", "Arial.ttf"],
        "Tahoma":  ["tahoma.ttf", "Tahoma.ttf"],
        "Calibri": ["calibri.ttf", "Calibri.ttf"],
    }
    for name, files in candidates.items():
        for fname in files:
            for d in search_dirs:
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
    c = canvas.Canvas(output_path)

    for page_data in tqdm(pages_data, desc="Tạo PDF"):
        img_array  = page_data["img"]
        blocks     = page_data["blocks"]
        w_pt, h_pt = page_data["page_size_pt"]

        c.setPageSize((w_pt, h_pt))

        # Lớp 1: ảnh scan gốc
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

            pdf_y     = h_pt - y2_pt
            font_size = max(box_h * 0.85, 4)
            text_w    = c.stringWidth(text, font_name, font_size)
            h_scale   = (box_w / text_w * 100) if text_w > 0 else 100

            t = c.beginText(x1_pt, pdf_y)
            t.setFont(font_name, font_size)
            t.setTextRenderMode(3)       # invisible, searchable
            t.setHorizScale(h_scale)
            t.textLine(text)
            c.drawText(t)

        c.showPage()

    c.save()
    print(f"[PDF] Lưu: {output_path}")


# ─────────────────────────────────────────────
# 7. LƯU / ĐỌC JSON
# ─────────────────────────────────────────────
def save_json(pages_data: list, output_path: str, source_pdf: str):
    out = {
        "document_id": Path(source_pdf).stem,
        "source_file": source_pdf,
        "total_pages": len(pages_data),
        "pages": [],
    }
    for i, pd in enumerate(pages_data):
        img_h, img_w = pd["img"].shape[:2]
        blocks = pd["blocks"]
        confs  = [b["confidence"] for b in blocks]
        out["pages"].append({
            "page_number":     i + 1,
            "image_size":      {"width": img_w, "height": img_h},
            "page_size_pt":    {"width": pd["page_size_pt"][0], "height": pd["page_size_pt"][1]},
            "confidence_page": round(sum(confs)/len(confs), 4) if confs else 0.0,
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
        {"img": img, "blocks": data["pages"][i]["text_blocks"], "page_size_pt": page_size_pt}
        for i, (img, page_size_pt) in enumerate(pages_imgs)
    ]


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    input_pdf:   str,
    output_dir:  str,
    dpi:         int  = 200,
    device:      str  = "gpu",
    font_dir:    str  = None,
    skip_ocr:    bool = False,
    beamsearch:  bool = False,
    batch_size:  int  = 16,
):
    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_pdf).stem
    out_pdf  = os.path.join(output_dir, f"{stem}_searchable.pdf")
    out_json = os.path.join(output_dir, f"{stem}_ocr.json")

    t_start = time.time()

    # Bước 1: Convert PDF → images
    pages_imgs = pdf_to_images(input_pdf, dpi=dpi)

    if skip_ocr and os.path.exists(out_json):
        pages_data = load_json_cache(out_json, pages_imgs)
    else:
        # ── Bước 2: DETECTION (PaddleOCR) ──
        paddle_device = "gpu" if device in ("gpu", "cuda") else "cpu"
        detector = init_paddle_detector(paddle_device)

        print(f"[DET] Đang detect {len(pages_imgs)} trang...\n")
        all_boxes = []
        for img, _ in tqdm(pages_imgs, desc="Detection"):
            boxes = detect_page(detector, img)
            all_boxes.append(boxes)

        # Giải phóng VRAM Paddle sau khi detect xong
        del detector
        import paddle
        paddle.device.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"\n[DET] Hoàn thành. Đã giải phóng VRAM Paddle.\n")

        # ── Bước 3: RECOGNITION (VietOCR) ──
        torch_device = "cuda:0" if device in ("gpu", "cuda") else "cpu"
        predictor = init_vietocr(torch_device, beamsearch=beamsearch)

        print(f"[REC] Đang nhận dạng chữ...\n")
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
                print(f"\n  ── Preview trang 1: {len(blocks)} đoạn ──")
                for b in blocks[:10]:
                    print(f"    {b['text'][:80]}")
                if len(blocks) > 10:
                    print(f"    ... và {len(blocks)-10} đoạn nữa\n")

        # Lưu JSON cache trước khi tạo PDF
        save_json(pages_data, out_json, input_pdf)

    # ── Bước 4: Tạo Searchable PDF ──
    font_name = register_font(font_dir)
    create_searchable_pdf(pages_data, out_pdf, font_name)

    elapsed = time.time() - t_start
    n = len(pages_imgs)
    print(f"\n{'='*55}")
    print(f"✅ Hoàn thành! {elapsed:.1f}s ({elapsed/n:.1f}s/trang)")
    print(f"   📄 Searchable PDF : {out_pdf}")
    print(f"   📋 JSON OCR       : {out_json}")
    print(f"{'='*55}\n")

    return out_pdf, out_json


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Pipeline v3: PaddleOCR det + VietOCR rec"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     default="output")
    parser.add_argument("--dpi",        type=int,  default=200)
    parser.add_argument("--device",     default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--font_dir",   default=None)
    parser.add_argument("--skip_ocr",   action="store_true",
                        help="Bỏ qua OCR, dùng JSON cache để tạo lại PDF")
    parser.add_argument("--beamsearch", action="store_true",
                        help="Bật beamsearch (chính xác hơn nhưng chậm ~3x)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Số text crops xử lý cùng lúc (mặc định 16 cho 4GB VRAM)")
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
