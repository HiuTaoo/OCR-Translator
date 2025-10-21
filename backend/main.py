import io
import hashlib
import asyncio
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from PIL import Image
from paddleocr import PaddleOCR
import cv2

from sklearn.cluster import DBSCAN
import numpy as np

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Config
# =========================

CONFIG = {
    "model_name": "facebook/m2m100_418M",
    "cache_size": 1000,
    "max_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ocr_gpu": torch.cuda.is_available(),
}

# =========================
# Caching
# =========================
class TranslationCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[str]:
        return self.cache.get(self._hash_text(text))

    def set(self, text: str, translation: str):
        if len(self.cache) >= self.max_size:
            # FIFO đơn giản
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[self._hash_text(text)] = translation

# =========================
# OCR
# =========================
class OCRProcessor:
    def __init__(self):
        # PaddleOCR CPU mode, không dùng angle classifier
        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            det=True,
            rec_batch_num=4,
            use_gpu=CONFIG["ocr_gpu"],
            rec=True,
            cls=False
        )

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image using PaddleOCR, safe for CPU mode"""
        try:
            # Validate ảnh
            h, w = image.shape[:2]
            if h < 10 or w < 10:
                logger.warning(f"Skip OCR: image too small ({w}x{h})")
                return []

            # Resize nếu quá nhỏ
            min_size = 32
            if h < min_size or w < min_size:
                scale = max(min_size / h, min_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h))
                logger.debug(f"Resized tiny image to ({new_w}x{new_h})")

            # Gọi OCR
            results = self.ocr.ocr(image)

            text_blocks: List[Dict] = []
            if results:
                # PaddleOCR thường trả về [ [ (coords, (text, conf)), ... ] ]
                page_results = results[0] if isinstance(results[0], list) else results

                for i, line in enumerate(page_results):
                    try:
                        if not line or len(line) < 2:
                            continue
                        coords, text_info = line[0], line[1]

                        # Lấy text + confidence
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = str(text_info[0]).strip(), float(text_info[1])
                        else:
                            text, confidence = str(text_info).strip(), 1.0

                        if not text or confidence < 0.3:
                            continue

                        # Tính bbox an toàn
                        bbox = {'x': 0, 'y': 0, 'width': w, 'height': h}
                        if coords and isinstance(coords, (list, tuple)) and len(coords) >= 4:
                            xs = [float(c[0]) for c in coords if len(c) >= 2]
                            ys = [float(c[1]) for c in coords if len(c) >= 2]
                            if xs and ys:
                                bbox = {
                                    'x': min(xs),
                                    'y': min(ys),
                                    'width': max(xs) - min(xs),
                                    'height': max(ys) - min(ys)
                                }

                        text_blocks.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })

                    except Exception as e:
                        logger.warning(f"Error parsing line {i}: {e}")
                        continue

            if text_blocks:
                text_blocks.sort(key=lambda x: (x['bbox']['y'], x['bbox']['x']))
            logger.info(f"OCR extracted {len(text_blocks)} text blocks")
            return text_blocks

        except Exception as e:
            logger.error(f"PaddleOCR Error: {str(e)}", exc_info=True)
            import paddleocr
            return [{
                'text': f"OCR Error: {str(e)}",
                'confidence': 0.0,
                'bbox': {'x': 0, 'y': 0, 'width': 300, 'height': 50},
                'debug': {
                    'error_type': type(e).__name__,
                    'paddleocr_version': getattr(paddleocr, '__version__', 'unknown'),
                    'image_shape': image.shape if hasattr(image, 'shape') else 'unknown'
                }
            }]


# =========================
# Translation
# =========================
class TranslationModel:
    def __init__(self, model_name: str, device: str):
        self.device = device
        logger.info(f"Loading translation model: {model_name}")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        if device == "cuda":
            self.model = self.model.to(device).half()
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Model loaded on {device}")

    def translate_batch(self, texts: List[str], src_lang: str = "en", tgt_lang: str = "vi") -> List[str]:
        if not texts:
            return []

        try:
            self.tokenizer.src_lang = src_lang
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

            if self.device == "cuda":
                # move to GPU nhưng chỉ half các tensor float
                for k, v in encoded.items():
                    if torch.is_floating_point(v):
                        encoded[k] = v.to(self.device).half()
                    else:
                        encoded[k] = v.to(self.device)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                    max_length=512,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )

            translations = [self.tokenizer.decode(t, skip_special_tokens=True).strip()
                            for t in generated_tokens]
            return translations
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return ["Translation failed"] * len(texts)

# =========================
# Globals
# =========================
translation_cache = TranslationCache(CONFIG["cache_size"])
ocr_processor = OCRProcessor()  # <-- KHÔNG truyền use_gpu
translation_model = None
executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])

# =========================
# FastAPI
# =========================
app = FastAPI(title="OCR Translation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cân nhắc siết trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global translation_model
    logger.info("Starting up OCR Translation API...")
    translation_model = TranslationModel(CONFIG["model_name"], CONFIG["device"])
    logger.info("API ready")

@app.on_event("shutdown")
async def shutdown_event():
    executor.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def group_text_blocks(text_blocks: List[Dict], max_distance: int = 120) -> List[Dict]:
    """
    Gom nhóm các text_blocks thành cụm (bubble) dựa trên vị trí gần nhau.
    Mỗi cụm trả về gồm:
      - 'text': đoạn hội thoại ghép lại
      - 'bbox': vùng bao quanh toàn bộ chữ trong cụm
    """
    if not text_blocks:
        return []

    # Tâm của từng box để gom cụm
    centers = np.array([
        [b['bbox']['x'] + b['bbox']['width'] / 2,
         b['bbox']['y'] + b['bbox']['height'] / 2]
        for b in text_blocks
    ])

    # Gom nhóm theo khoảng cách (DBSCAN)
    clustering = DBSCAN(eps=max_distance, min_samples=1).fit(centers)
    labels = clustering.labels_

    grouped = []
    for cluster_id in set(labels):
        cluster_blocks = [b for b, l in zip(text_blocks, labels) if l == cluster_id]
        if not cluster_blocks:
            continue

        # Phân biệt cụm dọc hoặc ngang
        vertical_count = sum(
            1 for b in cluster_blocks if b['bbox']['height'] > b['bbox']['width'] * 2
        )
        is_vertical = vertical_count > len(cluster_blocks) / 2

        # Sắp xếp text trong cụm
        if is_vertical:
            cluster_blocks.sort(key=lambda b: (b['bbox']['x'], b['bbox']['y']))
        else:
            cluster_blocks.sort(key=lambda b: (b['bbox']['y'], b['bbox']['x']))

        # Gộp text
        paragraph = ' '.join(b['text'] for b in cluster_blocks)

        # Tính bounding box bao toàn cụm
        x_min = min(b['bbox']['x'] for b in cluster_blocks)
        y_min = min(b['bbox']['y'] for b in cluster_blocks)
        x_max = max(b['bbox']['x'] + b['bbox']['width'] for b in cluster_blocks)
        y_max = max(b['bbox']['y'] + b['bbox']['height'] for b in cluster_blocks)

        grouped.append({
            'text': paragraph,
            'bbox': {
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            }
        })

    return grouped

@app.post("/ocr-translate")
async def ocr_translate(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # ==== Đọc và chuyển ảnh ====
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ==== Ghi nhận kích thước ảnh (dùng cho tính vị trí) ====
        img_h, img_w = cv_image.shape[:2]

        logger.info("Starting OCR...")
        text_blocks = await asyncio.get_event_loop().run_in_executor(
            executor, ocr_processor.extract_text, cv_image
        )

        if not text_blocks:
            return JSONResponse({
                "success": False,
                "message": "No text found in image",
                "original": "",
                "translated": "",
                "image_width": img_w,
                "image_height": img_h
            })

        # ==== Gom nhóm text thành cụm thoại ====
        paragraphs = group_text_blocks(text_blocks)

        # Ghép text để dịch
        original_text = '\n'.join(p['text'] for p in paragraphs)
        logger.info(f"Found {len(paragraphs)} text paragraphs")

        # ==== Cache dịch ====
        cached = translation_cache.get(original_text)
        if cached:
            logger.info("Using cached translation")
            return JSONResponse({
                "success": True,
                "original": original_text,
                "translated": cached,
                "paragraphs": paragraphs,
                "text_blocks": text_blocks,
                "image_width": img_w,
                "image_height": img_h
            })

        # ==== Dịch văn bản ====
        logger.info("Starting translation...")
        texts_to_translate = [p['text'] for p in paragraphs]

        translations = await asyncio.get_event_loop().run_in_executor(
            executor, translation_model.translate_batch, texts_to_translate
        )

        translated_text = '\n'.join(translations)
        translation_cache.set(original_text, translated_text)

        # ==== Trả kết quả ====
        return JSONResponse({
            "success": True,
            "original": original_text,
            "translated": translated_text,
            "paragraphs": paragraphs,
            "text_blocks": text_blocks,
            "image_width": img_w,
            "image_height": img_h
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ocr_translate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": CONFIG["device"],
        "cuda_available": torch.cuda.is_available(),
        "cache_size": len(translation_cache.cache)
    }

@app.post("/clear-cache")
async def clear_cache():
    translation_cache.cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=True
    )


