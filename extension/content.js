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

        // ==== Nếu backend trả về dữ liệu hợp lệ ====
        if (result.success && result.paragraphs && result.paragraphs.length > 0) {
            const main = result.paragraphs[0];
            const { x, y, width, height } = main.bbox;
            this.showResult(
                result.original,
                result.translated,
                x,
                y,
                width,
                height,
                result.image_width,
                result.image_height
            );
        } else {
            // fallback nếu không có bbox
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

    showResult(original, translated, left, top, width, height, imgW, imgH) {
    this.removeResultOverlay();

    const offset = 16;
    const boxWidth = 360;
    const boxHeight = 220;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const textCenter = left + width / 2;
    const preferRight = textCenter >= imgW / 2;
    const boxLeft = preferRight ? vw - boxWidth - offset : offset;

    const scaleY = vh / (imgH || vh);
    const boxTop = Math.max(offset, Math.min(top * scaleY, vh - boxHeight - offset));

    // ==== tạo popup ====
    this.resultOverlay = document.createElement('div');
    this.resultOverlay.id = 'ocr-translator-result';
    this.resultOverlay.style.cssText = `
        position: fixed;
        left: ${boxLeft}px;
        top: ${boxTop}px;
        width: ${boxWidth}px;
        background: white;
        border: 2px solid #007cba;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 1000000;
        font-family: Arial, sans-serif;
        font-size: 14px;
        max-height: 300px;
        overflow-y: auto;
    `;

    this.resultOverlay.innerHTML = `
        <div style="padding:8px 12px;border-bottom:1px solid #ddd;background:#f5f5f5;
                    display:flex;justify-content:space-between;align-items:center;">
            <strong>Translation Result</strong>
            <button id="ocr-close-btn" style="background:#ff4444;color:#fff;border:none;
                    border-radius:4px;padding:2px 8px;cursor:pointer;">×</button>
        </div>
        <div style="padding:12px;">
            <div style="margin-bottom:12px;">
                <strong>Original:</strong>
                <div style="background:#f9f9f9;padding:8px;border-radius:4px;
                            margin-top:4px;white-space:pre-wrap;">${original}</div>
            </div>
            <div>
                <strong style="color:#007cba;">Vietnamese:</strong>
                <div style="background:#e3f2fd;padding:8px;border-radius:4px;
                            margin-top:4px;white-space:pre-wrap;">${translated}</div>
            </div>
            <div style="margin-top:12px;text-align:center;">
                <button id="ocr-copy-btn" style="background:#007cba;color:#fff;border:none;
                        border-radius:4px;padding:6px 12px;cursor:pointer;margin-right:8px;">
                        Copy Translation</button>
                <button id="ocr-copy-all-btn" style="background:#28a745;color:#fff;border:none;
                        border-radius:4px;padding:6px 12px;cursor:pointer;">Copy Both</button>
            </div>
        </div>
    `;
    document.body.appendChild(this.resultOverlay);

    // ==== gán sự kiện cho nút ====
    document.getElementById('ocr-close-btn').onclick = () => this.removeResultOverlay();
    document.getElementById('ocr-copy-btn').onclick = () => this.copyToClipboard(translated);
    document.getElementById('ocr-copy-all-btn').onclick = () =>
        this.copyToClipboard(`Original: ${original}\n\nTranslated: ${translated}`);

    setTimeout(() => this.removeResultOverlay(), 30000);
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
        if (instruction) {
            instruction.remove();
        }
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
            this.showInstruction('Copied to clipboard!');
            setTimeout(() => this.hideInstruction(), 2000);
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