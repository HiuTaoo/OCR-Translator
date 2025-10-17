// Content script for OCR Translator Extension
if (window.__OCR_TRANSLATOR_LOADED__) {
    console.log("OCRTranslator already loaded");
    throw new Error("Duplicate load");
}
window.__OCR_TRANSLATOR_LOADED__ = true;

class OCRTranslator {
    constructor() {
        this.isSelecting = false;
        this.isProcessing = false;
        this.startX = 0;
        this.startY = 0;
        this.endX = 0;
        this.endY = 0;
        this.overlay = null;
        this.selectionBox = null;
        this.resultOverlay = null;

        // bind methods once
        this.onMouseDown = this.onMouseDown.bind(this);
        this.onMouseMove = this.onMouseMove.bind(this);
        this.onMouseUp = this.onMouseUp.bind(this);

        this.init();
    }

    init() {
        // Listen for messages from popup/background
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'startSelection') {
                this.startSelection();
                sendResponse({success: true});
            } else if (request.action === 'stopSelection') {
                this.stopSelection();
                sendResponse({success: true});
            }
        });

        // Listen for keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                this.startSelection();
            }
            if (e.key === 'Escape') {
                this.stopSelection();
            }
        });
    }

    startSelection() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        this.captureFullScreen();
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

        // ==== hiển thị nhiều popup tương ứng số đoạn thoại ====
        if (result.success && result.paragraphs && result.paragraphs.length > 0) {
            const translations = result.translated.split('\n');
            result.paragraphs.forEach((p, i) => {
                const text = p.text || '';
                const translated = translations[i] || '';
                const { x, y, width, height } = p.bbox;
                this.showResult(
                    text,
                    translated,
                    x,
                    y,
                    width,
                    height,
                    result.image_width,
                    result.image_height
                );
            });
        } else {
            this.showResult(result.original, result.translated, 20, 20, 400, 200, window.innerWidth, window.innerHeight);
        }
    } catch (e) {
        console.error(e);
        this.showError(e.message);
    } finally {
        this.isProcessing = false;
        this.hideInstruction();
    }
}

    stopSelection() {
        if (!this.isSelecting) return;
        this.isSelecting = false;

        // remove event listeners correctly
        document.removeEventListener('mousedown', this.onMouseDown);
        document.removeEventListener('mousemove', this.onMouseMove);
        document.removeEventListener('mouseup', this.onMouseUp);

        document.body.style.cursor = '';
        this.removeOverlay();
        this.hideInstruction();
}


    createOverlay() {
        if (this.overlay) return;

        this.overlay = document.createElement('div');
        this.overlay.id = 'ocr-translator-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.3);
            z-index: 999999;
            pointer-events: none;
        `;

        this.selectionBox = document.createElement('div');
        this.selectionBox.id = 'ocr-translator-selection';
        this.selectionBox.style.cssText = `
            position: absolute;
            border: 2px dashed #ff4444;
            background: rgba(255, 68, 68, 0.1);
            display: none;
        `;

        this.overlay.appendChild(this.selectionBox);
        document.body.appendChild(this.overlay);
    }

    removeOverlay() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
            this.selectionBox = null;
        }
    }

    onMouseDown(e) {
        if (!this.isSelecting) return;

        e.preventDefault();
        this.startX = e.clientX;
        this.startY = e.clientY;

        if (this.selectionBox) {
            this.selectionBox.style.display = 'block';
            this.selectionBox.style.left = this.startX + 'px';
            this.selectionBox.style.top = this.startY + 'px';
            this.selectionBox.style.width = '0px';
            this.selectionBox.style.height = '0px';
        }
    }

    onMouseMove(e) {
        if (!this.isSelecting || !this.selectionBox) return;

        this.endX = e.clientX;
        this.endY = e.clientY;

        const left = Math.min(this.startX, this.endX);
        const top = Math.min(this.startY, this.endY);
        const width = Math.abs(this.endX - this.startX);
        const height = Math.abs(this.endY - this.startY);

        this.selectionBox.style.left = left + 'px';
        this.selectionBox.style.top = top + 'px';
        this.selectionBox.style.width = width + 'px';
        this.selectionBox.style.height = height + 'px';
    }

    onMouseUp(e) {
        if (!this.isSelecting || this.isProcessing) return;

        this.endX = e.clientX;
        this.endY = e.clientY;

        const width = Math.abs(this.endX - this.startX);
        const height = Math.abs(this.endY - this.startY);

        // Minimum selection size
        if (width < 10 || height < 10) {
            this.stopSelection();
            return;
        }

        // Capture the selected area
        this.captureSelection();
    }

    async captureSelection() {
        if (this.isCapturing) return;

        if (this.isProcessing) return;
            this.isProcessing = true;

        this.isCapturing = true;
        setTimeout(() => { this.isCapturing = false; }, 1000);

        try {
            this.showInstruction('Processing image...');

            // Calculate selection bounds relative to viewport
            const left = Math.min(this.startX, this.endX);
            const top = Math.min(this.startY, this.endY);
            const width = Math.abs(this.endX - this.startX);
            const height = Math.abs(this.endY - this.startY);

            // Capture screenshot using Chrome API
            const canvas = await this.captureScreenshot(left, top, width, height);

            // Convert to blob
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/png');
            });

            // Send to backend
            const result = await this.sendToBackend(blob);

           if (result.success) {
              // Quyết định vị trí theo vùng chọn trên trang, không dùng bbox của OCR
               this.showResult(result.original, result.translated, left, top, width, height);
           } else {
               this.showError(result.message || 'Failed to process image');
           }

        } catch (error) {
            console.error('Capture error:', error);
            this.showError('Error capturing image: ' + error.message);
        } finally {
            this.stopSelection();
            this.isProcessing = false;
        }
    }

    async captureScreenshot(left, top, width, height) {
        return new Promise((resolve, reject) => {
            // Send message to background script to capture screenshot
            chrome.runtime.sendMessage({
                action: 'captureScreenshot',
                area: { left, top, width, height }
            }, (response) => {
                if (response.error) {
                    reject(new Error(response.error));
                    return;
                }

                // Create canvas from dataURL
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');

                    // Draw the cropped area
                    const scale = window.devicePixelRatio || 1;
                    ctx.drawImage(
                        img,
                        left * scale, top * scale,
                        width * scale, height * scale,
                        0, 0, width, height
                    );

                    resolve(canvas);
                };
                img.onerror = () => reject(new Error('Failed to load screenshot'));
                img.src = response.dataUrl;
            });
        });
    }

    async sendToBackend(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'screenshot.png');

        try {
            const response = await fetch('http://127.0.0.1:8000/ocr-translate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Backend error:', error);
            return {
                success: false,
                message: 'Backend connection failed: ' + error.message
            };
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

        const leftCss   = left   * scaleX;
        const topCss    = top    * scaleY;
        const widthCss  = width  * scaleX;
        const heightCss = height * scaleY;

        // Kiểm tra popup nên nằm bên trái hay phải vùng giữa
        const isLeftSide = (leftCss + widthCss / 2) > vw / 2;

        // Lấy vị trí cuộn trang hiện tại
        const scrollX = window.scrollX;
        const scrollY = window.scrollY;

        // Tính vị trí popup trong toàn trang (document)
        const boxLeft = (isLeftSide ? margin : (vw - boxWidth - margin)) + scrollX;
        const boxTop = Math.min(Math.max(topCss, margin), vh - boxHeight - margin) + scrollY;

        const overlay = document.createElement('div');
        overlay.id = `ocr-translator-result-${id}`;
        overlay.style.cssText = `
            position: absolute;  /* đổi từ fixed sang absolute để cuộn theo trang */
            left: ${boxLeft}px;
            top: ${boxTop}px;
            width: ${boxWidth}px;
            background: rgba(255,255,255,0.9);
            color: #111;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
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
                    text-shadow:none;
                ">×</button>
            </div>
            <div style="padding:8px;">
                <div style="margin-bottom:6px;">
                    <div style="
                        background:rgba(240,240,240,0.9);
                        color:#111;
                        padding:8px;
                        border-radius:4px;
                        white-space:pre-wrap;
                        text-shadow:0 1px 1px rgba(0,0,0,0.3);
                    ">${original}</div>
                </div>
                <div>
                    <div style="
                        background:rgba(200,230,255,0.9);
                        color:#000;
                        padding:8px;
                        border-radius:4px;
                        white-space:pre-wrap;
                        text-shadow:0 1px 1px rgba(255,255,255,0.2);
                    ">${translated}</div>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        overlay.querySelector('.ocr-close-btn').onclick = () => overlay.remove();

        // Xóa popup sau 30 giây
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

    removeResultOverlay() {
        if (this.resultOverlay) {
            this.resultOverlay.remove();
            this.resultOverlay = null;
        }
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showInstruction('Copied!');
            setTimeout(() => this.hideInstruction(), 1500);
        } catch (error) {
            console.error('Copy failed:', error);
            this.showInstruction('Copy failed', 'error');
            setTimeout(() => this.hideInstruction(), 3000);
        }
    }
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new OCRTranslator();
    });
} else {
    if (!window.__OCR_INSTANCE__) {
    window.__OCR_INSTANCE__ = new OCRTranslator();
    }
}