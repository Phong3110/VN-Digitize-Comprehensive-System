"""
OCR Script - Dễ debug, tham số cứng
Chỉnh sửa trực tiếp các biến trong phần CONFIG bên dưới
"""

import io
import json
import os
import re
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

# ══════════════════════════════════════════════
# CONFIG — chỉnh tại đây
# ══════════════════════════════════════════════
INPUT_PDF   = r"E:\project_cty_ocr\pdf_image_only.pdf"
OUTPUT_DIR  = r"E:\project_cty_ocr\output"
DPI         = 200          # 200 đủ tốt, tăng 300 nếu chữ nhỏ/mờ
BATCH_SIZE  = 16           # số dòng xử lý cùng lúc, giảm nếu OOM
BEAMSEARCH  = False        # True = chính xác hơn nhưng chậm ~3x
DEVICE_PADDLE = "gpu"      # "gpu" hoặc "cpu"
DEVICE_VIET   = "cuda:0"   # "cuda:0" hoặc "cpu"

# Tên file output
STEM     = Path(INPUT_PDF).stem
OUT_JSON = os.path.join(OUTPUT_DIR, f"{STEM}_ocr.json")
OUT_PDF  = os.path.join(OUTPUT_DIR, f"{STEM}_searchable.pdf")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# BƯỚC 1: ĐỌC PDF
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 1: Đọc PDF")
print("=" * 60)

doc   = fitz.open(INPUT_PDF)
zoom  = DPI / 72.0
mat   = fitz.Matrix(zoom, zoom)
pages = []

for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    pages.append({
        "img":          img,
        "page_size_pt": (page.rect.width, page.rect.height),
        "page_num":     i + 1,
    })

doc.close()
print(f"  → Đọc xong {len(pages)} trang. Kích thước trang 1: {pages[0]['img'].shape}\n")


# ══════════════════════════════════════════════
# BƯỚC 2: DETECTION (PaddleOCR)
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 2: Detection (PaddleOCR)")
print("=" * 60)

detector = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True,
    text_detection_model_name="PP-OCRv5_server_det",
    device=DEVICE_PADDLE,
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_det_thresh=0.3,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=2.0,
)

all_boxes = []
for p in tqdm(pages, desc="  Detection"):
    img    = p["img"]
    img_h, img_w = img.shape[:2]
    results = detector.predict(img)

    boxes = []
    for res in results:
        if res is None:
            continue
        for poly in res.get("rec_polys", []):
            polygon = poly.tolist() if hasattr(poly, "tolist") else list(poly)
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            boxes.append({
                "x1": int(min(xs)), "y1": int(min(ys)),
                "x2": int(max(xs)), "y2": int(max(ys)),
            })

    all_boxes.append(boxes)

print(f"\n  → Detection xong.")
print(f"  → Trang 1: {len(all_boxes[0])} boxes")
print(f"  → Trang 2: {len(all_boxes[1])} boxes")

# Giải phóng VRAM
del detector
try:
    import paddle
    paddle.device.cuda.empty_cache()
except Exception:
    pass
torch.cuda.empty_cache()
print("  → Đã giải phóng VRAM Paddle.\n")


# ══════════════════════════════════════════════
# BƯỚC 3: TÁCH DÒNG
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 3: Tách block → dòng đơn")
print("=" * 60)

def split_box_to_lines(box, img_w):
    """Tách box nhiều dòng thành các dòng đơn."""
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    box_h = y2 - y1
    estimated_line_h = max(20, int(img_w * 0.030))
    n_lines = max(1, round(box_h / estimated_line_h))

    if n_lines == 1:
        return [(x1, y1, x2, y2)]

    line_h = box_h // n_lines
    return [
        (x1, y1 + i * line_h, x2, y1 + (i+1) * line_h if i < n_lines-1 else y2)
        for i in range(n_lines)
    ]

def nms_boxes(boxes, overlap_thresh=0.5):
    """
    Loại bỏ các box chồng nhau (overlap > overlap_thresh).
    Ưu tiên giữ box có diện tích lớn hơn (box thật thường lớn hơn).
    """
    if not boxes:
        return []

    # Sắp xếp theo y1 (từ trên xuống)
    boxes = sorted(boxes, key=lambda b: b[1])
    keep = []

    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue

        dominated = False
        for kx1, ky1, kx2, ky2 in keep:
            # Tính phần chồng nhau
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            # Overlap ratio so với box nhỏ hơn
            k_area = (kx2 - kx1) * (ky2 - ky1)
            overlap = inter / min(area, k_area)
            if overlap > overlap_thresh:
                dominated = True
                break

        if not dominated:
            keep.append(box)

    return keep

# Tách dòng cho tất cả trang + NMS
all_line_boxes = []
for i, (p, boxes) in enumerate(zip(pages, all_boxes)):
    img_w = p["img"].shape[1]

    # Tách block → dòng đơn
    line_boxes_raw = []
    for box in boxes:
        line_boxes_raw.extend(split_box_to_lines(box, img_w))

    # NMS: loại bỏ box chồng nhau
    line_boxes_clean = nms_boxes(line_boxes_raw, overlap_thresh=0.5)

    all_line_boxes.append(line_boxes_clean)

print(f"  → Trang 1: {len(all_boxes[0])} blocks → {len(all_line_boxes[0])} dòng (sau NMS)")
print(f"  → Trang 2: {len(all_boxes[1])} blocks → {len(all_line_boxes[1])} dòng (sau NMS)\n")


# ══════════════════════════════════════════════
# BƯỚC 4: RECOGNITION (VietOCR)
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 4: Recognition (VietOCR)")
print("=" * 60)

config = Cfg.load_config_from_name("vgg_transformer")
config["device"]                  = DEVICE_VIET
config["predictor"]["beamsearch"] = BEAMSEARCH
predictor = Predictor(config)

def is_valid_text(text: str) -> bool:
    """Lọc bỏ text vô nghĩa: toàn số dài, không có chữ cái."""
    text = text.strip()
    if len(text) < 2:
        return False
    if re.fullmatch(r'[\d\s\-\.\,\/]+', text) and len(text) > 6:
        return False
    if not re.search(r'[a-zA-ZÀ-ỹ]', text):
        return False
    return True

PAD = 3
all_results = []

for i, (p, line_boxes) in enumerate(tqdm(
    zip(pages, all_line_boxes), total=len(pages), desc="  Recognition"
)):
    img   = p["img"]
    img_h, img_w = img.shape[:2]

    # Crop từng dòng
    crops = []
    valid_boxes = []
    for (x1, y1, x2, y2) in line_boxes:
        cx1 = max(0, x1 - PAD);  cy1 = max(0, y1 - PAD)
        cx2 = min(img_w, x2 + PAD); cy2 = min(img_h, y2 + PAD)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crops.append(Image.fromarray(img[cy1:cy2, cx1:cx2]))
        valid_boxes.append((x1, y1, x2, y2))

    if not crops:
        all_results.append([])
        continue

    # VietOCR theo batch
    texts = []
    for b in range(0, len(crops), BATCH_SIZE):
        texts.extend(predictor.predict_batch(crops[b:b + BATCH_SIZE]))

    # Ghép kết quả + filter
    blocks = []
    for (x1, y1, x2, y2), text in zip(valid_boxes, texts):
        text = unicodedata.normalize("NFC", text.strip())
        if not is_valid_text(text):
            continue
        blocks.append({
            "text":       text,
            "confidence": 0.95 if len(text) >= 3 else 0.60,
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "low_confidence": len(text) < 3,
        })

    all_results.append(blocks)

    # Preview trang 1
    if i == 0:
        print(f"\n  ── Preview trang 1: {len(blocks)} dòng sau filter ──")
        for b in blocks[:15]:
            print(f"    {b['text']}")
        print()

print(f"\n  → Recognition xong.\n")


# ══════════════════════════════════════════════
# BƯỚC 5: LƯU JSON
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 5: Lưu JSON")
print("=" * 60)

output_json = {
    "document_id": STEM,
    "source_file": INPUT_PDF,
    "total_pages": len(pages),
    "pages":       [],
}

for i, (p, blocks) in enumerate(zip(pages, all_results)):
    img_h, img_w = p["img"].shape[:2]
    confs = [b["confidence"] for b in blocks]
    output_json["pages"].append({
        "page_number":     i + 1,
        "image_size":      {"width": img_w, "height": img_h},
        "page_size_pt":    {"width": p["page_size_pt"][0], "height": p["page_size_pt"][1]},
        "confidence_page": round(sum(confs)/len(confs), 4) if confs else 0.0,
        "text_blocks":     blocks,
        "full_text":       " ".join(b["text"] for b in blocks),
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)

print(f"  → Đã lưu: {OUT_JSON}\n")


# ══════════════════════════════════════════════
# BƯỚC 6: TẠO SEARCHABLE PDF
# ══════════════════════════════════════════════
print("=" * 60)
print("BƯỚC 6: Tạo Searchable PDF")
print("=" * 60)

# Đăng ký font
font_name = "Helvetica"
for fname in ["arial.ttf", "tahoma.ttf", "calibri.ttf"]:
    fpath = os.path.join(r"C:\Windows\Fonts", fname)
    if os.path.exists(fpath):
        try:
            pdfmetrics.registerFont(TTFont("VietFont", fpath))
            font_name = "VietFont"
            print(f"  → Font: {fpath}")
            break
        except Exception:
            pass

c = canvas.Canvas(OUT_PDF)

for p, blocks in tqdm(zip(pages, all_results), total=len(pages), desc="  Tạo PDF"):
    img_array  = p["img"]
    w_pt, h_pt = p["page_size_pt"]
    img_h, img_w = img_array.shape[:2]

    c.setPageSize((w_pt, h_pt))

    # Lớp 1: ảnh scan
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=85)
    buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, width=w_pt, height=h_pt)

    # Lớp 2: invisible text
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
        t.setTextRenderMode(3)
        t.setHorizScale(h_scale)
        t.textLine(text)
        c.drawText(t)

    c.showPage()

c.save()
print(f"\n  → Đã lưu: {OUT_PDF}\n")

print("=" * 60)
print("✅ HOÀN THÀNH!")
print(f"   📋 JSON : {OUT_JSON}")
print(f"   📄 PDF  : {OUT_PDF}")
print("=" * 60)
