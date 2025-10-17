
        class PopupController {
            constructor() {
                this.backendStatus = false;
                this.settings = {
                    autoClose: true,
                    showOriginal: true
                };
                
                this.init();
            }
            
            async init() {
                await this.loadSettings();
                this.setupEventListeners();
                this.checkBackendStatus();
                
                // Check backend status every 30 seconds
                setInterval(() => this.checkBackendStatus(), 30000);
            }
            
            async loadSettings() {
                try {
                    const result = await chrome.storage.sync.get(['autoClose', 'showOriginal']);
                    this.settings.autoClose = result.autoClose !== false;
                    this.settings.showOriginal = result.showOriginal !== false;
                    
                    this.updateToggleUI();
                } catch (error) {
                    console.error('Failed to load settings:', error);
                }
            }
            
            setupEventListeners() {
                const startBtn = document.getElementById('start-btn');
                const toggleAutoClose = document.getElementById('toggle-autoclose');
                const toggleOriginal = document.getElementById('toggle-original');
                
                startBtn.addEventListener('click', () => {
                    this.startSelection();
                });
                
                toggleAutoClose.addEventListener('click', () => {
                    this.toggleSetting('autoClose');
                });
                
                toggleOriginal.addEventListener('click', () => {
                    this.toggleSetting('showOriginal');
                });
            }
            
            async checkBackendStatus() {
                const statusElement = document.getElementById('status');
                const statusText = document.getElementById('status-text');
                const startBtn = document.getElementById('start-btn');
                
                statusText.innerHTML = '<div class="loading"></div>Checking backend...';
                
                try {
                    const response = await chrome.runtime.sendMessage({action: 'checkBackend'});
                    
                    if (response.available) {
                        this.backendStatus = true;
                        statusElement.className = 'status online';
                        statusText.textContent = `Backend online (${response.status.device})`;
                        startBtn.disabled = false;
                        startBtn.textContent = 'Translate Visible Area';
                    } else {
                        throw new Error(response.error);
                    }
                } catch (error) {
                    this.backendStatus = false;
                    statusElement.className = 'status offline';
                    statusText.textContent = 'Backend offline - Start server first';
                    startBtn.disabled = true;
                    startBtn.textContent = 'Backend Not Available';
                }
            }
            
            async startSelection() {
                if (!this.backendStatus) {
                    alert('Backend server is not running. Please start the Python server first.');
                    return;
                }
                
                try {
                    await chrome.runtime.sendMessage({action: 'startSelection'});
                    window.close(); // Close popup after starting selection
                } catch (error) {
                    console.error('Failed to start selection:', error);
                    alert('Failed to start selection. Please try again.');
                }
            }
            
            async toggleSetting(setting) {
                this.settings[setting] = !this.settings[setting];
                
                try {
                    await chrome.storage.sync.set({[setting]: this.settings[setting]});
                    this.updateToggleUI();
                } catch (error) {
                    console.error('Failed to save setting:', error);
                    // Revert the change
                    this.settings[setting] = !this.settings[setting];
                }
            }
            
            updateToggleUI() {
                const toggleAutoClose = document.getElementById('toggle-autoclose');
                const toggleOriginal = document.getElementById('toggle-original');
                
                toggleAutoClose.className = this.settings.autoClose ? 'toggle active' : 'toggle';
                toggleOriginal.className = this.settings.showOriginal ? 'toggle active' : 'toggle';
            }
        }
        
        // Initialize popup when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            new PopupController();
        });
