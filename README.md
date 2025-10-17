# OCR Translator Setup Instructions

## Cấu trúc thư mục

```
ocr-translator/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── config.py (optional)
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── content.js
│   ├── content.css
│   ├── background.js
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
└── README.md
```

## 1. Cài đặt Backend

### Bước 1: Tạo môi trường Python

```bash
cd backend
python -m venv venv

# Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
venv\Scripts\activate

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý cho RTX3050:**
- Đảm bảo CUDA toolkit đã được cài đặt
- Nếu gặp lỗi với PaddlePaddle GPU, thử:
```bash
pip install paddlepaddle-gpu==2.5.1 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

### Bước 3: Test backend

```bash
python main.py
```

Backend sẽ chạy tại `http://127.0.0.1:8000`

Kiểm tra health endpoint:
```bash
curl http://127.0.0.1:8000/health
```

## 2. Cài đặt Extension

### Bước 1: Tạo icons

Tạo thư mục `extension/icons/` và thêm các file icon:
- `icon16.png` (16x16px)
- `icon48.png` (48x48px)  
- `icon128.png` (128x128px)

Hoặc sử dụng icon đơn giản:
```bash
# Tạo icon đơn giản bằng Python
python -c "
from PIL import Image, ImageDraw
import os

os.makedirs('extension/icons', exist_ok=True)

for size in [16, 48, 128]:
    img = Image.new('RGBA', (size, size), (0, 124, 186, 255))
    draw = ImageDraw.Draw(img)
    
    # Vẽ chữ T
    if size >= 48:
        font_size = size // 3
        draw.text((size//2-font_size//2, size//2-font_size//2), 'T', 
                 fill='white', anchor='mm')
    
    img.save(f'extension/icons/icon{size}.png')

print('Icons created!')
"
```

### Bước 2: Load extension vào Chrome

1. Mở Chrome/Edge
2. Vào `chrome://extensions/`
3. Bật "Developer mode" 
4. Click "Load unpacked"
5. Chọn thư mục `extension/`

## 3. Sử dụng

### Khởi động hệ thống

1. **Start backend:**
```bash
cd backend
python main.py
```

2. **Sử dụng extension:**
   - Click icon extension hoặc nhấn `Ctrl+Shift+S`
   - Chọn vùng ảnh bằng chuột
   - Đợi kết quả dịch

### Phím tắt

- `Ctrl+Shift+S`: Bắt đầu chọn vùng
- `Escape`: Hủy chọn
- Click vào kết quả để copy

## 4. Tối ưu cho RTX3050 + i5-11400H + 16GB RAM

### Backend optimizations

Sửa file `main.py`:

```python
CONFIG = {
    "model_name": "facebook/m2m100_418M",  # Model nhỏ hơn
    "cache_size": 500,  # Giảm cache size
    "max_workers": 2,   # Tối đa 2 workers
    "device": "cuda",
    "ocr_gpu": True,
    "batch_size": 4,    # Batch nhỏ
}
```

### Cấu hình PaddleOCR

```python
self.ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    show_log=False,
    use_space_char=True,
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    max_text_length=25
)
```

## 5. Xử lý lỗi thường gặp

### Backend không khởi động được

```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu CUDA không available, sửa config:
CONFIG["device"] = "cpu"
CONFIG["ocr_gpu"] = False
```

### Extension không hoạt động

1. Kiểm tra Developer Console (F12)
2. Reload extension tại `chrome://extensions/`
3. Kiểm tra permissions trong manifest.json

### OCR không chính xác

- Chọn vùng ảnh rõ nét hơn
- Thử với ảnh có độ phân giải cao hơn
- Điều chỉnh threshold trong PaddleOCR config

### Dịch thuật chậm

- Giảm batch_size
- Sử dụng model nhỏ hơn: `facebook/m2m100_418M`
- Tăng cache_size để tránh dịch lại

## 6. Mở rộng tính năng

### Thêm model dịch khác

Sửa `main.py`:

```python
# Thay M2M100 bằng NLLB
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

### Tự động detect speech bubbles

Thêm vào `main.py`:

```python
import cv2

def detect_speech_bubbles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum bubble size
            x, y, w, h = cv2.boundingRect(contour)
            bubbles.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    return bubbles
```

### Export kết quả

Thêm endpoint vào `main.py`:

```python
@app.post("/export")
async def export_results(data: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"translation_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {"filename": filename}
```

## 7. Production Deployment

### Tối ưu backend cho production

```python
# Thêm vào main.py
import logging
from contextlib import asynccontextmanager

# Disable debug logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Production config
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1", 
        port=8000,
        reload=False,
        workers=1,  # Single worker cho RTX3050
        log_level="info"
    )
```

### Package extension

```bash
# Tạo .zip file cho Chrome Web Store
cd extension
zip -r ../ocr-translator-extension.zip . -x "*.DS_Store" "*.git*"
```

## Troubleshooting

### Memory issues

```python
# Thêm vào main.py
import gc
import torch

@app.middleware("http")
async def cleanup_memory(request, call_next):
    response = await call_next(request)
    
    # Clean up GPU memory after each request
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return response
```

### Performance monitoring

```python
import time
import psutil

@app.get("/stats")
async def get_stats():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    }
```