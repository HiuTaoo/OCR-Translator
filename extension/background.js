// Background script for OCR Translator Extension

// Handle extension commands (shortcuts)
chrome.commands.onCommand.addListener((command) => {
    if (command === 'start_selection') {
        startSelection();
    }
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'captureScreenshot') {
        captureScreenshot(request.area, sendResponse);
        return true; // Keep message channel open for async response
    } else if (request.action === 'startSelection') {
        startSelection();
        sendResponse({success: true});
    } else if (request.action === 'checkBackend') {
        checkBackendStatus(sendResponse);
        return true; // Keep message channel open for async response
    }
    else if (request.action === 'captureFullPage') {
        captureScreenshot(null, sendResponse); // không cần area
        return true;
    }

});

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
    startSelection();
});

async function startSelection() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) return;

        // Kiểm tra nếu content.js đã inject
        const [{ result } = {}] = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => !!window.__OCR_TRANSLATOR_LOADED__,
        });

        if (!result) {
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['content.js'],
            });
        } else {
            console.log('OCRTranslator already present, skip reinjection');
        }

        chrome.tabs.sendMessage(tab.id, { action: 'startSelection' }, (response) => {
            if (chrome.runtime.lastError) {
                console.error('Error sending message:', chrome.runtime.lastError.message);
            }
        });

    } catch (error) {
        console.error('Error starting selection:', error);
    }
}


async function captureScreenshot(area, sendResponse) {
    try {
        const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
        
        if (!tab) {
            sendResponse({error: 'No active tab found'});
            return;
        }

        // Capture the visible tab
        const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
            format: 'png',
            quality: 100
        });

        sendResponse({dataUrl: dataUrl});
        
    } catch (error) {
        console.error('Screenshot capture error:', error);
        sendResponse({error: error.message});
    }
}

async function checkBackendStatus(sendResponse) {
    try {
        const response = await fetch('http://127.0.0.1:8000/health');
        const data = await response.json();
        sendResponse({
            available: true,
            status: data
        });
    } catch (error) {
        sendResponse({
            available: false,
            error: error.message
        });
    }
}

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
    console.log('OCR Translator Extension installed');
    
    // Set default settings
    chrome.storage.sync.set({
        backendUrl: 'http://127.0.0.1:8000',
        autoClose: true,
        showOriginal: true
    });
});