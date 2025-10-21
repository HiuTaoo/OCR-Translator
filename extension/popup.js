class PopupController {
    constructor() {
        this.backendStatus = false;
        this.extensionEnabled = false;
        this.init();
    }

    async init() {
        this.toggleBtn = document.getElementById('toggle-extension');
        this.startBtn = document.getElementById('start-btn');
        this.statusText = document.getElementById('status-text');
        this.statusBox = document.getElementById('status');
        this.inputSelect = document.getElementById('input-lang');
        this.outputSelect = document.getElementById('output-lang');

        await this.loadState();
        this.checkBackendStatus();
        this.setupEventListeners();

        setInterval(() => this.checkBackendStatus(), 30000);
    }

    async loadState() {
        const res = await chrome.storage.local.get(['extensionEnabled', 'inputLang', 'outputLang']);
        this.extensionEnabled = res.extensionEnabled || false;

        // ✅ hiển thị ngôn ngữ hiện tại đã lưu
        if (res.inputLang) this.inputSelect.value = res.inputLang;
        if (res.outputLang) this.outputSelect.value = res.outputLang;

        this.updateToggleUI();
    }

    setupEventListeners() {
        this.toggleBtn.addEventListener('click', () => this.toggleExtension());
        this.startBtn.addEventListener('click', () => this.startSelection());

        // ✅ lưu lại ngay khi đổi ngôn ngữ
        this.inputSelect.addEventListener('change', async () => {
            await chrome.storage.local.set({ inputLang: this.inputSelect.value });
        });
        this.outputSelect.addEventListener('change', async () => {
            await chrome.storage.local.set({ outputLang: this.outputSelect.value });
        });
    }

    async toggleExtension() {
        this.extensionEnabled = !this.extensionEnabled;
        await chrome.storage.local.set({ extensionEnabled: this.extensionEnabled });
        this.updateToggleUI();

        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab?.id) return;

        if (this.extensionEnabled) {
            try {
                await chrome.tabs.sendMessage(tab.id, { action: 'applyExtensionState', enabled: true });
            } catch {
                try {
                    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
                    await new Promise(r => setTimeout(r, 200));
                    await chrome.tabs.sendMessage(tab.id, { action: 'applyExtensionState', enabled: true });
                } catch {
                    alert('Unable to activate on this tab.');
                }
            }
        } else {
            const tabs = await chrome.tabs.query({});
            for (const t of tabs) {
                try {
                    await chrome.tabs.sendMessage(t.id, { action: 'applyExtensionState', enabled: false });
                } catch {}
            }
        }
    }

    async startSelection() {
        if (!this.backendStatus) {
            alert('Backend server not running. Start it first.');
            return;
        }

        const inputLang = this.inputSelect.value;
        const outputLang = this.outputSelect.value;

        try {
            await chrome.storage.local.set({ inputLang, outputLang });
            await chrome.runtime.sendMessage({ action: 'startSelection' });
            window.close();
        } catch {
            alert('Failed to start selection.');
        }
    }

    async checkBackendStatus() {
        this.statusText.innerHTML = '<span style="font-style:italic;">Checking...</span>';
        try {
            const res = await chrome.runtime.sendMessage({ action: 'checkBackend' });
            if (res.available) {
                this.backendStatus = true;
                this.statusBox.className = 'status online';
                this.statusText.textContent = `Backend Online (${res.status.device})`;
                this.startBtn.disabled = false;
            } else throw new Error();
        } catch {
            this.backendStatus = false;
            this.statusBox.className = 'status offline';
            this.statusText.textContent = 'Backend Offline — start server first';
            this.startBtn.disabled = true;
        }
    }

    updateToggleUI() {
        if (this.extensionEnabled) {
            this.toggleBtn.classList.add('active');
            this.toggleBtn.textContent = 'Extension Enabled';
        } else {
            this.toggleBtn.classList.remove('active');
            this.toggleBtn.textContent = 'Enable Extension';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => new PopupController());
