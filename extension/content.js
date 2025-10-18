if (window.__OCR_TRANSLATOR_LOADED__) {
    console.log("OCRTranslator already loaded");
    throw new Error("Duplicate load");
}
window.__OCR_TRANSLATOR_LOADED__ = true;

class OCRTranslator {
    constructor() {
        this.isProcessing = false;
        this.init();
    }

    init() {
        // Nhận lệnh từ popup hoặc phím tắt
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'startSelection') {
                this.startSelection();
                sendResponse({ success: true });
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                this.startSelection();
            }
        });
    }

    async startSelection() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        await this.captureFullScreen();
        this.isProcessing = false;
    }

    async captureFullScreen() {
        try {
            const response = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({ action: 'captureFullPage' }, (res) => {
                    if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
                    else resolve(res);
                });
            });

            if (response.error) throw new Error(response.error);

            const img = new Image();
            img.src = response.dataUrl;
            await img.decode();

            const canvas = document.createElement('canvas');
            canvas.width = window.innerWidth * devicePixelRatio;
            canvas.height = window.innerHeight * devicePixelRatio;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            const result = await this.sendToBackend(blob);

            if (result.success && result.paragraphs?.length > 0) {
                const translations = result.translated.split('\n');
                result.paragraphs.forEach((p, i) => {
                    const text = p.text || '';
                    const translated = translations[i] || '';
                    const { x, y, width, height } = p.bbox;
                    this.showResult(text, translated, x, y, width, height, result.image_width, result.image_height);
                });
            } else {
                this.showResult(result.original, result.translated, 20, 20, 400, 200, window.innerWidth, window.innerHeight);
            }
        } catch (e) {
            console.error(e);
            this.showError(e.message);
        } finally {
            this.hideInstruction();
        }
    }

    async sendToBackend(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'screenshot.png');

        try {
            const response = await fetch('http://127.0.0.1:8000/ocr-translate', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            return await response.json();
        } catch (error) {
            console.error('Backend error:', error);
            return { success: false, message: 'Backend connection failed: ' + error.message };
        }
    }

    showResult(original, translated, left, top, width, height, imgW, imgH, id = Date.now() + Math.random()) {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const boxWidth = 360;
        const boxHeight = 220;
        const margin = 8;

        const scaleX = vw / imgW;
        const scaleY = vh / imgH;

        const leftCss = left * scaleX;
        const topCss = top * scaleY;
        const widthCss = width * scaleX;
        const heightCss = height * scaleY;

        const isLeftSide = (leftCss + widthCss / 2) < vw / 2;

        const scrollContainer = document.scrollingElement || document.documentElement;
        const scrollX = scrollContainer.scrollLeft;
        const scrollY = scrollContainer.scrollTop;

        const boxLeft = isLeftSide
            ? margin + scrollX
            : vw - boxWidth - margin + scrollX;
        const boxTop = Math.min(Math.max(topCss, margin), vh - boxHeight - margin) + scrollY;

        const overlay = document.createElement('div');
        overlay.id = `ocr-translator-result-${id}`;
        overlay.style.cssText = `
            position: absolute;
            left: ${boxLeft}px;
            top: ${boxTop}px;
            width: ${boxWidth}px;
            background: rgba(255,255,255,0.9);
            color: #111;
            border: 1px solid rgba(0,0,0,0.2);
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            z-index: 1000000;
            font-family: Arial, sans-serif;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
            backdrop-filter: blur(4px);
        `;

        overlay.innerHTML = `
            <div style="
                display:flex;
                align-items:center;
                justify-content:space-between;
                padding:6px 10px;
                border-bottom:1px solid rgba(0,0,0,0.15);
                background:rgba(0,0,0,0.05);
                border-radius:8px 8px 0 0;
            ">
                <button class="ocr-close-btn" style="
                    background:#ff4444;
                    color:#fff;
                    border:none;
                    border-radius:4px;
                    padding:2px 8px;
                    cursor:pointer;
                    font-weight:bold;
                    font-size:14px;
                    line-height:1;
                ">×</button>
            </div>
            <div style="padding:8px;">
                <div style="margin-bottom:6px;">
                    <div style="
                        background:rgba(240,240,240,0.9);
                        padding:8px;
                        border-radius:4px;
                        white-space:pre-wrap;
                    ">${original}</div>
                </div>
                <div>
                    <div style="
                        background:rgba(200,230,255,0.9);
                        padding:8px;
                        border-radius:4px;
                        white-space:pre-wrap;
                    ">${translated}</div>
                </div>
            </div>
        `;

        scrollContainer.appendChild(overlay);
        overlay.querySelector('.ocr-close-btn').onclick = () => overlay.remove();
        setTimeout(() => overlay.remove(), 30000);
    }

    showError(message) {
        this.showInstruction(`Error: ${message}`, 'error');
        setTimeout(() => this.hideInstruction(), 5000);
    }

    showInstruction(text, type = 'info') {
        this.hideInstruction();
        const instruction = document.createElement('div');
        instruction.id = 'ocr-translator-instruction';
        instruction.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'error' ? '#ff4444' : '#007cba'};
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 1000001;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        `;
        instruction.textContent = text;
        document.body.appendChild(instruction);
    }

    hideInstruction() {
        const instruction = document.getElementById('ocr-translator-instruction');
        if (instruction) instruction.remove();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new OCRTranslator());
} else {
    if (!window.__OCR_INSTANCE__) window.__OCR_INSTANCE__ = new OCRTranslator();
}
