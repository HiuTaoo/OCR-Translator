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
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'startSelection') {
                this.startSelection();
                sendResponse({success: true});
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

        if (!(await this.isExtensionEnabled())) {
            console.warn('Extension is disabled — OCR blocked');
            this.showInstruction('Extension is OFF — cannot OCR.', 'error');
            return;
        }

        this.isProcessing = true;
        await this.captureFullScreen();
        this.isProcessing = false;
    }

    async isExtensionEnabled() {
        return new Promise((resolve) => {
            chrome.storage.local.get(['extensionEnabled'], (res) => {
                resolve(res.extensionEnabled === true);
            });
        });
    }


    async captureFullScreen() {
        try {
            const response = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({action: 'captureFullPage'}, (res) => {
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
                    const {x, y, width, height} = p.bbox;
                    this.showResult(text, translated, x, y, width, height, result.image_width, result.image_height);
                });
            } else {
                console.log('No OCR text detected — skip popup');
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
            return {success: false, message: 'Backend connection failed: ' + error.message};
        }
    }

    getScrollableParent(el = document.body) {
        let node = el;
        while (node && node !== document) {
            const style = getComputedStyle(node);
            const overflowY = style.overflowY;
            if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
                return node;
            }
            node = node.parentNode;
        }
        return document.scrollingElement || document.documentElement;
    }

    showResult(original, translated, left, top, width, height, imgW, imgH, id = Date.now() + Math.random()) {
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const boxHeight = 220;
        const margin = 8;

        const leftZone = document.querySelector('body > div[style*="left: 0px"]');
        const rightZone = document.querySelector('body > div[style*="right: 0px"]');
        const leftZoneWidth = leftZone ? leftZone.offsetWidth : 200;
        const rightZoneWidth = rightZone ? rightZone.offsetWidth : 200;

        const scaleX = vw / imgW;
        const scaleY = vh / imgH;
        const leftCss = left * scaleX;
        const topCss = top * scaleY;
        const widthCss = width * scaleX;
        const heightCss = height * scaleY;

        const isLeftSide = (leftCss + widthCss / 2) < vw / 2;

        const scrollContainer = this.getScrollableParent(document.body);
        const scrollX = scrollContainer.scrollLeft;
        const scrollY = scrollContainer.scrollTop;

        const zone = isLeftSide ? leftZone : rightZone;
        const zoneWidth = isLeftSide ? leftZoneWidth : rightZoneWidth;

        const boxLeft = isLeftSide
            ? scrollX + margin
            : vw - zoneWidth + scrollX + margin;

        const boxTop = Math.min(Math.max(topCss, margin), vh - boxHeight - margin) + scrollY;
        const boxWidth = zoneWidth - 2 * margin;

        const overlay = document.createElement('div');
        overlay.id = `ocr-translator-result-${id}`;
        overlay.style.cssText = `
            position: absolute;
            left: ${boxLeft}px;
            top: ${boxTop}px;
            width: ${boxWidth}px;
            max-width: ${boxWidth}px;
            background: #ffffff;      
            color: #000000;              
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 1000000;
            font-family: Arial, sans-serif;
            font-size: 16px;
            max-height: 300px;
            overflow: hidden;
            padding: 10px;
            line-height: 1.5;
            transition: background 0.2s, color 0.2s, width 0.2s;
            cursor: default;
            user-select: text;
            box-sizing: border-box;
        `;

        overlay.innerHTML = `
            <div class="ocr-text" style="
                white-space: pre-wrap;
                word-wrap: break-word;
                color: #000;
                text-align: center;
            ">${translated}</div>
        `;
        overlay.addEventListener('mouseenter', () => {
            overlay.querySelector('.ocr-text').textContent = original;
            overlay.style.background = '#f0f0f0';
            overlay.style.color = '#333';
        });
        overlay.addEventListener('mouseleave', () => {
            overlay.querySelector('.ocr-text').textContent = translated;
            overlay.style.background = '#ffffff';
            overlay.style.color = '#000000';
        });


        scrollContainer.appendChild(overlay);

        // ✅ Theo dõi khi side zone thay đổi kích thước
        if (zone) {
            const observer = new ResizeObserver(() => {
                const newWidth = zone.offsetWidth - 2 * margin;
                overlay.style.width = `${newWidth}px`;
                overlay.style.maxWidth = `${newWidth}px`;
                if (!isLeftSide) {
                    overlay.style.left = `${vw - zone.offsetWidth + scrollX + margin}px`;
                }
            });
            observer.observe(zone);
            overlay._observer = observer;
        }

        overlay.addEventListener('click', () => {
            overlay._observer?.disconnect();
            overlay.remove();
        });
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

        setTimeout(() => {
            instruction.style.opacity = '0';
            setTimeout(() => instruction.remove(), 400);
        }, 2500);
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

// ==========================
// Adjustable side zones
// ==========================

(function handleSideZones() {
    let leftZone = null, rightZone = null;
    let widths = {left: 200, right: 200};
    let enabled = false;

    chrome.runtime.onMessage.addListener((req) => {
        if (req.action === 'applyExtensionState') {
            enabled = req.enabled;
            if (enabled) createZones();
            else removeZones();
        }
    });

    function createZones() {
        removeZones();
        leftZone = makeZone('left', widths.left);
        rightZone = makeZone('right', widths.right);
        document.body.append(leftZone, rightZone);
        makeResizable(leftZone, 'left');
        makeResizable(rightZone, 'right');
        updateColor();
    }

    function removeZones() {
        leftZone?.remove();
        rightZone?.remove();
        leftZone = rightZone = null;

        const existingPopups = document.querySelectorAll('[id^="ocr-translator-result-"]');
        existingPopups.forEach(el => el.remove());
    }

    function makeZone(side, width) {
        const zone = document.createElement('div');
        Object.assign(zone.style, {
            position: 'fixed',
            top: '0',
            bottom: '0',
            width: width + 'px',
            [side]: '0',
            zIndex: '999999',
            cursor: 'ew-resize',
            background: '#333333'
        });
        return zone;
    }

    function makeResizable(zone, side) {
        const resizer = document.createElement('div');
        Object.assign(resizer.style, {
            position: 'absolute',
            top: '0',
            bottom: '0',
            width: '6px',
            background: 'rgba(255,0,0,0.47)',
            cursor: 'col-resize',
            [side === 'left' ? 'right' : 'left']: '0'
        });
        zone.appendChild(resizer);

        resizer.addEventListener('mousedown', (e) => {
            e.preventDefault();
            document.body.style.cursor = 'col-resize';
            const startX = e.clientX;
            const startWidth = zone.offsetWidth;

            const move = (ev) => {
                const delta = ev.clientX - startX;
                const newWidth = side === 'left'
                    ? Math.max(50, startWidth + delta)
                    : Math.max(50, startWidth - delta);
                zone.style.width = newWidth + 'px';
                widths[side] = newWidth;
            };
            const up = () => {
                document.body.style.cursor = 'default';
                document.removeEventListener('mousemove', move);
                document.removeEventListener('mouseup', up);
            };
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', up);
        });
    }

    function updateColor() {
        const color = enabled ? '#333333' : 'transparent';
        if (leftZone) leftZone.style.background = color;
        if (rightZone) rightZone.style.background = color;
    }
})();


