/**
 * Ghostline Creative Intelligence System - Main Application JavaScript
 * Complete version with submit functionality fixes
 */

'use strict';

// =============================================================================
// GLOBAL VARIABLES AND STATE MANAGEMENT
// =============================================================================

let isSubmitting = false;
let attachedFiles = [];

// Chat history management configuration
const chatHistoryManager = {
    maxVisibleMessages: 50,
    fadeThreshold: 10,
    showingRecentOnly: false
};

// Global thread sidebar instance
let threadSidebar;

// =============================================================================
// TIMEZONE DETECTION
// =============================================================================

function initializeTimezoneDetection() {
    // Only run if user is logged in and we have the API endpoints
    if (!document.body.classList.contains('logged-in')) {
        return;
    }
    
    // Check if we've already detected timezone this session
    const alreadyDetected = sessionStorage.getItem('timezone_detected');
    if (alreadyDetected === 'true') {
        return;
    }
    
    // Detect browser timezone
    const browserTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    
    if (!browserTimezone) {
        console.warn('Could not detect browser timezone');
        return;
    }
    
    // Send to server
    fetch('/api/timezone/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify({
            timezone: browserTimezone
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Timezone auto-detected:', browserTimezone);
            sessionStorage.setItem('timezone_detected', 'true');
            showTimezoneNotification(browserTimezone);
        } else {
            console.warn('Timezone detection failed:', data.error);
        }
    })
    .catch(error => {
        console.warn('Timezone detection request failed:', error);
    });
}

function showTimezoneNotification(timezone) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #059669;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>🌍</span>
            <span>Timezone detected: ${timezone.split('/')[1]?.replace('_', ' ') || timezone}</span>
            <button onclick="this.parentElement.parentElement.remove()" 
                    style="background: none; border: none; color: white; cursor: pointer; margin-left: 10px; font-size: 18px;">×</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Fade in
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 100);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentElement) {
                notification.parentElement.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// =============================================================================
// CHAT HISTORY MANAGEMENT
// =============================================================================

function toggleHistoryView() {
    const messages = document.querySelectorAll('.message');
    const toggleButton = document.getElementById('historyToggle');
    
    if (!toggleButton || messages.length === 0) return;
    
    if (chatHistoryManager.showingRecentOnly) {
        // Show all messages
        messages.forEach(msg => {
            msg.classList.remove('fade-out');
            msg.classList.remove('archived');
        });
        
        chatHistoryManager.showingRecentOnly = false;
        toggleButton.textContent = 'Show Recent Only';
        
        console.log('Showing all messages');
    } else {
        // Show only recent messages
        const messagesToArchive = Math.max(0, messages.length - chatHistoryManager.maxVisibleMessages);
        const messagesToFade = Math.min(chatHistoryManager.fadeThreshold,
                                       Math.max(0, messages.length - chatHistoryManager.maxVisibleMessages + chatHistoryManager.fadeThreshold));
        
        // Archive oldest messages
        for (let i = 0; i < messagesToArchive; i++) {
            messages[i].classList.add('archived');
        }
        
        // Fade messages near the threshold
        for (let i = messagesToArchive; i < messagesToArchive + messagesToFade; i++) {
            if (messages[i]) {
                messages[i].classList.add('fade-out');
            }
        }
        
        chatHistoryManager.showingRecentOnly = true;
        toggleButton.textContent = `Show All Messages (${messages.length})`;
        
        console.log(`Showing recent only: archived ${messagesToArchive}, faded ${messagesToFade}`);
    }
    
    // Scroll to show recent messages
    setTimeout(() => {
        scrollToBottom();
    }, 100);
}

function clearOldMessages() {
    const messages = document.querySelectorAll('.message');
    const keepCount = 100; // Keep last 100 messages
    
    if (messages.length <= keepCount) {
        alert('No old messages to clear - you have fewer than 100 messages.');
        return;
    }
    
    const messagesToRemove = messages.length - keepCount;
    const confirmed = confirm(
        `This will permanently remove ${messagesToRemove} old messages, keeping only the most recent ${keepCount}. ` +
        `This action cannot be undone. Continue?`
    );
    
    if (!confirmed) return;
    
    // Remove old messages from DOM
    for (let i = 0; i < messagesToRemove; i++) {
        if (messages[i] && !messages[i].id) { // Don't remove the bottom anchor
            messages[i].remove();
        }
    }
    
    // Reset history manager state
    chatHistoryManager.showingRecentOnly = false;
    const toggleButton = document.getElementById('historyToggle');
    if (toggleButton) {
        toggleButton.textContent = 'Show Recent Only';
    }
    
    // Show confirmation
    const notification = document.createElement('div');
    notification.className = 'bookmark-notification';
    notification.textContent = `✅ Cleared ${messagesToRemove} old messages`;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
    
    console.log(`Cleared ${messagesToRemove} old messages, kept ${keepCount} recent messages`);
}

function autoManageChatHistory() {
    const messages = document.querySelectorAll('.message:not(.archived)');
    
    // Only auto-manage if we have more messages than the threshold
    if (messages.length <= chatHistoryManager.maxVisibleMessages + chatHistoryManager.fadeThreshold) {
        return;
    }
    
    // Don't auto-manage if user is explicitly viewing all messages
    if (!chatHistoryManager.showingRecentOnly) {
        return;
    }
    
    const excessMessages = messages.length - chatHistoryManager.maxVisibleMessages;
    
    // Fade out excess messages gradually
    for (let i = 0; i < Math.min(excessMessages, chatHistoryManager.fadeThreshold); i++) {
        if (messages[i] && !messages[i].classList.contains('fade-out')) {
            messages[i].classList.add('fade-out');
        }
    }
    
    // Archive messages that are far beyond the threshold
    const archiveThreshold = chatHistoryManager.maxVisibleMessages + chatHistoryManager.fadeThreshold + 10;
    if (messages.length > archiveThreshold) {
        const messagesToArchive = messages.length - archiveThreshold;
        for (let i = 0; i < messagesToArchive; i++) {
            if (messages[i] && !messages[i].classList.contains('archived')) {
                messages[i].classList.add('archived');
            }
        }
    }
    
    // Update toggle button text if it exists
    const toggleButton = document.getElementById('historyToggle');
    if (toggleButton && chatHistoryManager.showingRecentOnly) {
        const totalMessages = document.querySelectorAll('.message').length;
        toggleButton.textContent = `Show All Messages (${totalMessages})`;
    }
}

// =============================================================================
// IMAGE HANDLER
// =============================================================================

const imageHandler = {
    handleResponseData: function(messageElement, responseDataStr) {
        try {
            const responseData = JSON.parse(responseDataStr);
            
            if (responseData && responseData.image_data && responseData.image_url) {
                this.addInlineImage(messageElement, responseData.image_data, responseData.image_url);
            }
        } catch (error) {
            console.error('Failed to parse response data for images:', error);
        }
    },
    
    addInlineImage: function(messageElement, imageData, imageUrl) {
        const messageContent = messageElement.querySelector('.message-content');
        if (!messageContent) return;
        
        const imageContainer = document.createElement('div');
        imageContainer.style.marginTop = '15px';
        
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imageData}`;
        img.className = 'inline-generated-image';
        img.onclick = () => this.showImageModal(img.src);
        
        const actionButtons = document.createElement('div');
        actionButtons.className = 'image-action-buttons';
        
        actionButtons.innerHTML = `
            <button class="image-btn" onclick="imageHandler.downloadImage('${imageData}', 'generated-image.png')">
                💾 Download
            </button>
            <button class="image-btn secondary" onclick="imageHandler.copyImageToClipboard('${imageData}')">
                📋 Copy
            </button>
            <a href="${imageUrl}" target="_blank" class="image-btn success">
                🔗 Open Full Size
            </a>
        `;
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(actionButtons);
        messageContent.appendChild(imageContainer);
    },
    
    showImageModal: function(src) {
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.innerHTML = `<img src="${src}" alt="Generated image">`;
        modal.onclick = () => modal.remove();
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 10);
    },
    
    downloadImage: function(imageData, filename) {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${imageData}`;
        link.download = filename;
        link.click();
    },
    
    copyImageToClipboard: function(imageData) {
        // Convert base64 to blob and copy to clipboard
        fetch(`data:image/png;base64,${imageData}`)
            .then(res => res.blob())
            .then(blob => {
                navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
                alert('Image copied to clipboard!');
            })
            .catch(err => {
                console.error('Failed to copy image:', err);
                alert('Failed to copy image to clipboard');
            });
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function copyToClipboard(text, buttonElement) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = buttonElement.innerHTML;
        buttonElement.innerHTML = '✅ Copied';
        buttonElement.classList.add('success');
        
        setTimeout(() => {
            buttonElement.innerHTML = originalText;
            buttonElement.classList.remove('success');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        const originalText = buttonElement.innerHTML;
        buttonElement.innerHTML = '✅ Copied';
        setTimeout(() => {
            buttonElement.innerHTML = originalText;
        }, 2000);
    });
}

function speakText(text, buttonElement) {
    if ('speechSynthesis' in window) {
        // Stop any current speech
        speechSynthesis.cancel();
        
        if (buttonElement.classList.contains('playing')) {
            buttonElement.classList.remove('playing');
            return;
        }
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        
        buttonElement.classList.add('playing');
        
        utterance.onend = () => {
            buttonElement.classList.remove('playing');
        };
        
        utterance.onerror = () => {
            buttonElement.classList.remove('playing');
        };
        
        speechSynthesis.speak(utterance);
    } else {
        alert('Text-to-speech not supported in this browser');
    }
}

function recordFeedback(responseId, feedbackType, buttonElement) {
    // Visual feedback
    const feedbackButtons = buttonElement.parentElement.querySelectorAll('.feedback-btn');
    feedbackButtons.forEach(btn => btn.classList.remove('selected'));
    buttonElement.classList.add('selected');
    
    // Send feedback to server
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            response_id: responseId,
            feedback_type: feedbackType,
            timestamp: new Date().toISOString()
        })
    }).then(response => {
        if (response.ok) {
            console.log('Feedback recorded:', feedbackType);
        }
    }).catch(error => {
        console.error('Feedback failed:', error);
    });
}

// =============================================================================
// MENU HANDLING
// =============================================================================

function initializeMenu() {
    console.log('Initializing menu...');
    
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const menuOverlay = document.getElementById('menuOverlay');
    
    if (!hamburgerBtn || !dropdownMenu || !menuOverlay) {
        console.error('Menu elements not found');
        return;
    }
    
    // Fix hamburger button click handler
    hamburgerBtn.addEventListener('click', (e) => {
        console.log('Hamburger button clicked');
        e.preventDefault();
        e.stopPropagation();
        toggleMenu();
    });
    
    // Fix menu overlay click handler
    menuOverlay.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeMenu();
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdownMenu.contains(e.target) && !hamburgerBtn.contains(e.target)) {
            closeMenu();
        }
    });
    
    // Fix menu item click handlers
    dropdownMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        
        const menuItem = e.target.closest('.menu-item');
        if (menuItem) {
            console.log('Menu item clicked:', menuItem);
            
            if (menuItem.tagName === 'A') {
                console.log('Following link:', menuItem.href);
                setTimeout(() => closeMenu(), 100);
            } else if (menuItem.tagName === 'BUTTON') {
                console.log('Button clicked');
                
                const onclickAttr = menuItem.getAttribute('onclick');
                if (onclickAttr) {
                    try {
                        eval(onclickAttr);
                    } catch (error) {
                        console.error('Error executing onclick:', error);
                    }
                }
                
                setTimeout(() => closeMenu(), 100);
            }
        }
    });
    
    // Keyboard support
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeMenu();
        }
    });
    
    // Enhanced keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.altKey && !e.ctrlKey && !e.shiftKey) {
            switch(e.key.toLowerCase()) {
                case 'm':
                    e.preventDefault();
                    sendQuickCommand('good morning');
                    break;
                case 'e':
                    e.preventDefault();
                    sendQuickCommand('overnight');
                    break;
                case 'c':
                    e.preventDefault();
                    sendQuickCommand('calendar');
                    break;
                case 'r':
                    e.preventDefault();
                    sendQuickCommand('remind me to follow up in 1 hour');
                    break;
            }
        }
    });
    
    console.log('Menu initialized successfully with fixed click handlers');
}

function toggleMenu() {
    const dropdownMenu = document.getElementById('dropdownMenu');
    const isOpen = dropdownMenu.classList.contains('show');
    
    console.log('Toggle menu - currently open:', isOpen);
    
    if (isOpen) {
        closeMenu();
    } else {
        openMenu();
    }
}

function openMenu() {
    console.log('Opening menu');
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const menuOverlay = document.getElementById('menuOverlay');
    
    if (hamburgerBtn) hamburgerBtn.classList.add('active');
    if (dropdownMenu) dropdownMenu.classList.add('show');
    if (menuOverlay) menuOverlay.classList.add('show');
    
    // Ensure menu items are properly interactive
    const menuItems = dropdownMenu.querySelectorAll('.menu-item');
    menuItems.forEach(item => {
        item.style.pointerEvents = 'auto';
        item.style.cursor = 'pointer';
    });
}

function closeMenu() {
    console.log('Closing menu');
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const menuOverlay = document.getElementById('menuOverlay');
    
    if (hamburgerBtn) hamburgerBtn.classList.remove('active');
    if (dropdownMenu) dropdownMenu.classList.remove('show');
    if (menuOverlay) menuOverlay.classList.remove('show');
}

function sendQuickCommand(command) {
    console.log('Sending quick command:', command);
    const promptInput = document.getElementById('promptInput');
    
    if (promptInput) {
        promptInput.value = command;
        closeMenu();
        setTimeout(() => {
            submitForm();
        }, 100);
    }
}

function refreshPage() {
    location.reload();
}

// =============================================================================
// FILE HANDLING
// =============================================================================

function initializeFileHandling() {
    console.log('Initializing file handling...');
    
    const paperclipBtn = document.getElementById('paperclipBtn');
    const fileInput = document.getElementById('fileInput');
    const composerContainer = document.getElementById('composerContainer');
    const dropOverlay = document.getElementById('dropOverlay');

    if (!paperclipBtn || !fileInput || !composerContainer || !dropOverlay) {
        console.error('File handling elements not found');
        return;
    }

    // Paperclip button click
    paperclipBtn.addEventListener('click', () => {
        console.log('Paperclip button clicked');
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop on composer
    composerContainer.addEventListener('dragover', handleDragOver);
    composerContainer.addEventListener('drop', handleDrop);
    composerContainer.addEventListener('dragenter', handleDragEnter);
    composerContainer.addEventListener('dragleave', handleDragLeave);

    // Global drag and drop
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    document.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
            dropOverlay.classList.add('show');
        }
    });

    document.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!dropOverlay.contains(e.relatedTarget)) {
            dropOverlay.classList.remove('show');
        }
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropOverlay.classList.remove('show');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect({ target: { files: e.dataTransfer.files } });
        }
    });
    
    console.log('File handling initialized successfully');
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect({ target: { files: e.dataTransfer.files } });
    }
}

function handleFileSelect(e) {
    console.log('File select triggered');
    const files = Array.from(e.target.files);
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];

    for (const file of files) {
        // Check file size
        if (file.size > maxSize) {
            alert(`File "${file.name}" is too large. Maximum size is 10MB.`);
            continue;
        }

        // Check file type
        const isAllowed = allowedTypes.some(type => file.type.startsWith(type)) ||
                         file.name.toLowerCase().endsWith('.docx');
        
        if (!isAllowed) {
            alert(`File type not supported: ${file.name}. Please use images, PDFs, or Word documents.`);
            continue;
        }

        // Check if already attached
        if (attachedFiles.find(f => f.name === file.name && f.size === file.size)) {
            continue;
        }

        attachedFiles.push(file);
    }

    updateAttachedFilesUI();
    updatePaperclipButton();
}

function removeAttachedFile(index) {
    attachedFiles.splice(index, 1);
    updateAttachedFilesUI();
    updatePaperclipButton();
}

function updateAttachedFilesUI() {
    const container = document.getElementById('attachedFiles');
    
    if (!container) return;
    
    if (attachedFiles.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'flex';
    container.innerHTML = attachedFiles.map((file, index) => {
        const size = formatFileSize(file.size);
        const icon = getFileIcon(file);
        
        return `
            <div class="attached-file">
                <div class="file-icon">${icon}</div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${size}</div>
                </div>
                <button type="button" class="remove-file-btn" onclick="removeAttachedFile(${index})" title="Remove file">
                    ×
                </button>
            </div>
        `;
    }).join('');
}

function updatePaperclipButton() {
    const paperclipBtn = document.getElementById('paperclipBtn');
    if (!paperclipBtn) return;
    
    if (attachedFiles.length > 0) {
        paperclipBtn.classList.add('has-file');
        paperclipBtn.title = `${attachedFiles.length} file(s) attached`;
    } else {
        paperclipBtn.classList.remove('has-file');
        paperclipBtn.title = 'Attach files';
    }
}

function getFileIcon(file) {
    if (file.type.startsWith('image/')) return '🖼️';
    if (file.type === 'application/pdf') return '📄';
    if (file.type.includes('word') || file.name.toLowerCase().endsWith('.docx')) return '📝';
    return '📎';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// =============================================================================
// THREAD SIDEBAR FUNCTIONALITY
// =============================================================================

class ThreadSidebar {
    constructor() {
        this.sidebar = null;
        this.overlay = null;
        this.isOpen = false;
        this.currentThreadId = null;
        this.threads = [];
        this.searchTerm = '';
        
        this.init();
    }
    
    init() {
        this.createSidebarHTML();
        this.attachEventListeners();
        this.loadThreads();
        
        // Auto-refresh threads every 30 seconds
        setInterval(() => {
            this.loadThreads();
        }, 30000);
    }
    
    createSidebarHTML() {
        // Create sidebar overlay for mobile
        this.overlay = document.createElement('div');
        this.overlay.className = 'sidebar-overlay';
        this.overlay.addEventListener('click', () => this.close());
        document.body.appendChild(this.overlay);
        
        // Create sidebar
        this.sidebar = document.createElement('div');
        this.sidebar.className = 'thread-sidebar';
        this.sidebar.innerHTML = `
            <div class="sidebar-header">
                <div class="sidebar-title">
                    <span>📚</span>
                    Bookmarked Threads
                </div>
                <input type="text" class="sidebar-search" placeholder="Search bookmarks..." id="threadSearch">
            </div>
            <div class="thread-list" id="threadList">
                <div class="thread-loading">Loading bookmarks...</div>
            </div>
        `;
        
        document.body.appendChild(this.sidebar);
        
        // Add sidebar toggle to header
        this.addSidebarToggle();
    }
    
    addSidebarToggle() {
        const headerRight = document.querySelector('.header-right');
        if (headerRight) {
            const toggle = document.createElement('button');
            toggle.className = 'sidebar-toggle';
            toggle.innerHTML = '📚';
            toggle.title = 'Toggle Bookmarks';
            toggle.addEventListener('click', () => this.toggle());
            
            // Insert before the hamburger menu
            headerRight.insertBefore(toggle, headerRight.firstChild);
        }
    }
    
    attachEventListeners() {
        const searchInput = document.getElementById('threadSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchTerm = e.target.value.toLowerCase();
                this.renderThreads();
            });
        }
        
        // Close sidebar on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768 && this.isOpen) {
                document.body.classList.add('sidebar-open');
            } else {
                document.body.classList.remove('sidebar-open');
            }
        });
    }
    
    async loadThreads() {
        try {
            const response = await fetch('/api/threads?include_archived=false&limit=50', {
                credentials: 'include'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.threads = data.threads.filter(thread =>
                    thread.message_count > 0
                );
                this.renderThreads();
            } else {
                this.showError('Failed to load bookmarks');
            }
        } catch (error) {
            console.error('Failed to load threads:', error);
            this.showError('Connection error');
        }
    }
    
    renderThreads() {
        const threadList = document.getElementById('threadList');
        if (!threadList) return;
        
        let filteredThreads = this.threads;
        
        // Apply search filter
        if (this.searchTerm) {
            filteredThreads = this.threads.filter(thread =>
                thread.title.toLowerCase().includes(this.searchTerm) ||
                (thread.project && thread.project.toLowerCase().includes(this.searchTerm))
            );
        }
        
        if (filteredThreads.length === 0) {
            threadList.innerHTML = this.searchTerm ?
                `<div class="thread-empty">
                    <div class="thread-empty-icon">🔍</div>
                    <div>No bookmarks match "${this.searchTerm}"</div>
                </div>` :
                `<div class="thread-empty">
                    <div class="thread-empty-icon">📚</div>
                    <div>No bookmarked conversations yet</div>
                    <div style="font-size: 0.8rem; margin-top: 8px; opacity: 0.7;">
                        Say "bookmark this" to save conversations
                    </div>
                </div>`;
            return;
        }
        
        threadList.innerHTML = filteredThreads.map(thread => this.renderThreadItem(thread)).join('');
        
        // Attach click listeners
        threadList.querySelectorAll('.thread-item').forEach(item => {
            item.addEventListener('click', () => {
                const threadId = item.dataset.threadId;
                this.selectThread(threadId);
            });
        });
    }
    
    renderThreadItem(thread) {
        const isActive = thread.thread_id === this.currentThreadId;
        const timeAgo = this.formatTimeAgo(thread.last_message_at || thread.updated_at);
        const hasBookmarks = thread.message_count > 0;
        
        return `
            <div class="thread-item ${isActive ? 'active' : ''}" data-thread-id="${thread.thread_id}">
                <div class="thread-title">${this.escapeHtml(thread.title)}</div>
                <div class="thread-preview">${this.escapeHtml(thread.project || 'General')}</div>
                <div class="thread-meta">
                    <div class="thread-time">${timeAgo}</div>
                    <div class="thread-indicators">
                        ${hasBookmarks ? '<div class="bookmark-indicator">📌</div>' : ''}
                        <div class="message-count">${thread.message_count}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    async selectThread(threadId) {
        if (!threadId) return;
        
        try {
            // Show loading state
            const threadItems = document.querySelectorAll('.thread-item');
            threadItems.forEach(item => item.classList.remove('active'));
            
            const selectedItem = document.querySelector(`[data-thread-id="${threadId}"]`);
            if (selectedItem) {
                selectedItem.classList.add('active');
            }
            
            // Load thread content
            const response = await fetch(`/api/threads/${threadId}`, {
                credentials: 'include'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.currentThreadId = threadId;
                this.loadThreadIntoChat(data);
                
                // Close sidebar on mobile after selection
                if (window.innerWidth <= 768) {
                    this.close();
                }
            } else {
                this.showError('Failed to load thread');
            }
        } catch (error) {
            console.error('Failed to load thread:', error);
            this.showError('Failed to load conversation');
        }
    }
    
    loadThreadIntoChat(threadData) {
        const chatThread = document.getElementById('thread');
        const bottomAnchor = document.getElementById('bottom-anchor');
        
        if (!chatThread || !bottomAnchor) {
            console.error('Chat elements not found');
            return;
        }
        
        // Clear current messages (except the anchor)
        const messages = chatThread.querySelectorAll('.message');
        messages.forEach(msg => msg.remove());
        
        // Add thread messages
        if (threadData.conversations && threadData.conversations.length > 0) {
            threadData.conversations.forEach(conversation => {
                this.addMessageToChat(conversation, threadData.bookmarks);
            });
        }
        
        // Scroll to bottom
        setTimeout(() => {
            bottomAnchor.scrollIntoView({ behavior: 'smooth' });
        }, 100);
        
        // Show notification
        this.showNotification(`Loaded: ${threadData.metadata?.title || 'Thread'}`);
    }
    
    addMessageToChat(conversation, bookmarks = []) {
        const chatThread = document.getElementById('thread');
        const bottomAnchor = document.getElementById('bottom-anchor');
        
        if (!chatThread || !bottomAnchor) return;
        
        // Check if this conversation is bookmarked
        const isBookmarked = bookmarks.some(bookmark =>
            bookmark.chat_id === conversation.id
        );
        
        // Create user message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = `message user ${isBookmarked ? 'bookmarked' : ''}`;
        userMessageDiv.innerHTML = `
            <div class="message-bubble user">
                <div class="message-header">You</div>
                <div class="message-content">${this.escapeHtml(conversation.user_input)}</div>
            </div>
        `;
        
        chatThread.insertBefore(userMessageDiv, bottomAnchor);
        
        // Create AI response if exists
        if (conversation.response_data) {
            const responseData = conversation.response_data;
            
            Object.entries(responseData).forEach(([voice, response]) => {
                if (response && typeof response === 'string') {
                    const aiMessageDiv = document.createElement('div');
                    aiMessageDiv.className = `message bot ${isBookmarked ? 'bookmarked' : ''}`;
                    aiMessageDiv.innerHTML = `
                        <div class="message-bubble bot">
                            <div class="message-header">
                                <img src="/static/syntax-buffering.png" alt="${voice}" class="logo">
                                ${voice}
                                <button class="speaker-btn" onclick="speakText('${this.escapeForJs(response)}', this)" title="Speak this message">
                                    🔊
                                </button>
                            </div>
                            <div class="message-content">${this.escapeHtml(response)}</div>
                            <button class="mobile-copy-btn" onclick="copyToClipboard('${this.escapeForJs(response)}', this)" title="Copy this response">
                                📋 Copy
                            </button>
                        </div>
                    `;
                    
                    chatThread.insertBefore(aiMessageDiv, bottomAnchor);
                }
            });
        }
    }

    formatTimeAgo(dateString) {
        if (!dateString) return 'Unknown';
        
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString();
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }
    
    escapeForJs(text) {
        return (text || '').replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, '\\n');
    }
    
    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'bookmark-notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    showError(message) {
        const threadList = document.getElementById('threadList');
        if (threadList) {
            threadList.innerHTML = `
                <div class="thread-empty">
                    <div class="thread-empty-icon">⚠️</div>
                    <div style="color: #dc2626;">${message}</div>
                    <button onclick="threadSidebar.loadThreads()" style="margin-top: 8px; padding: 4px 8px; background: var(--surface-hover); border: 1px solid var(--border); border-radius: 4px; color: inherit; cursor: pointer;">
                        Retry
                    </button>
                </div>
            `;
        }
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.isOpen = true;
        this.sidebar.classList.add('open');
        
        if (window.innerWidth > 768) {
            document.body.classList.add('sidebar-open');
        } else {
            this.overlay.classList.add('show');
        }
        
        // Update toggle button
        const toggle = document.querySelector('.sidebar-toggle');
        if (toggle) {
            toggle.classList.add('active');
        }
        
        // Refresh threads when opening
        this.loadThreads();
    }
    
    close() {
        this.isOpen = false;
        this.sidebar.classList.remove('open');
        this.overlay.classList.remove('show');
        document.body.classList.remove('sidebar-open');
        
        // Update toggle button
        const toggle = document.querySelector('.sidebar-toggle');
        if (toggle) {
            toggle.classList.remove('active');
        }
    }
}

// =============================================================================
// BOOKMARK AND EXPORT UTILITIES
// =============================================================================

function detectBookmarkCommand(userInput) {
    const bookmarkPatterns = [
        /^bookmark$/i,
        /^bookmark this$/i,
        /^bookmark this as/i,
        /^save this$/i,
        /^mark this$/i
    ];
    
    return bookmarkPatterns.some(pattern => pattern.test(userInput.trim()));
}

function showBookmarkNotification(success = true, message = null) {
    const text = message || (success ? '✅ Bookmark created!' : '❌ Bookmark failed');
    
    const notification = document.createElement('div');
    notification.className = 'bookmark-notification';
    notification.textContent = text;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
    
    // Refresh sidebar threads if bookmark was successful
    if (success && window.threadSidebar) {
        setTimeout(() => {
            window.threadSidebar.loadThreads();
        }, 1000);
    }
}

function showExportProgress(message) {
    let progressElement = document.querySelector('.export-progress');
    
    if (!progressElement) {
        progressElement = document.createElement('div');
        progressElement.className = 'export-progress';
        progressElement.innerHTML = `
            <div class="export-progress-header">
                <span>📤</span>
                <span id="exportProgressText">Preparing export...</span>
            </div>
            <div class="export-progress-bar">
                <div class="export-progress-fill" id="exportProgressFill"></div>
            </div>
        `;
        document.body.appendChild(progressElement);
    }
    
    const progressText = document.getElementById('exportProgressText');
    const progressFill = document.getElementById('exportProgressFill');
    
    progressText.textContent = message;
    progressElement.classList.add('show');
    
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 30;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 300);
    
    return {
        update: (newMessage, newProgress = null) => {
            progressText.textContent = newMessage;
            if (newProgress !== null) {
                clearInterval(interval);
                progressFill.style.width = newProgress + '%';
            }
        },
        complete: (finalMessage = 'Export complete!') => {
            clearInterval(interval);
            progressText.textContent = finalMessage;
            progressFill.style.width = '100%';
            
            setTimeout(() => {
                progressElement.classList.remove('show');
                setTimeout(() => {
                    if (progressElement.parentNode) {
                        progressElement.parentNode.removeChild(progressElement);
                    }
                }, 300);
            }, 2000);
        }
    };
}

// =============================================================================
// FORM SUBMISSION AND CHAT - FIXED VERSION
// =============================================================================

async function submitForm() {
    console.log('Submit form called');
    
    if (isSubmitting) {
        console.log('Already submitting, ignoring');
        return;
    }
    
    const promptInput = document.getElementById('promptInput');
    const projectSelect = document.getElementById('projectSelect');
    
    if (!promptInput || !projectSelect) {
        console.error('Form elements not found');
        return;
    }
    
    const inputValue = promptInput.value.trim();
    
    // Allow submission if there's text OR files attached
    if (!inputValue && (!attachedFiles || attachedFiles.length === 0)) {
        promptInput.focus();
        return;
    }
    
    // Check for bookmark command
    const isBookmarkCommand = detectBookmarkCommand(inputValue);
    
    // Check for export command
    const isExportCommand = /export|copy to google docs|save to drive/i.test(inputValue);
    
    isSubmitting = true;
    
    const sendButton = document.getElementById('sendBtn');
    if (sendButton) {
        sendButton.disabled = true;
        sendButton.style.opacity = '0.5';
        sendButton.style.pointerEvents = 'none';
    }
    
    let exportProgress = null;
    
    try {
        // Show export progress for export commands
        if (isExportCommand) {
            exportProgress = showExportProgress('Preparing export to Google Drive...');
        }
        
        // If files are attached, handle file upload + analysis
        if (attachedFiles && attachedFiles.length > 0) {
            await handleFileUploadAndAnalysis(inputValue, projectSelect.value);
        } else {
            // Regular text-only chat
            const voices = getSelectedVoices();
            const random = document.querySelector('input[name="random"]')?.checked || false;
            
            // Add user message to chat
            addUserMessage(inputValue);
            promptInput.value = '';
            autoResize(promptInput);
            
            // Start streaming chat
            await startStreamingChat(inputValue, projectSelect.value, voices, random, {
                isBookmarkCommand,
                isExportCommand,
                exportProgress
            });
        }
    } catch (error) {
        console.error('Chat failed:', error);
        addErrorMessage('Failed to send message: ' + error.message);
        
        if (exportProgress) {
            exportProgress.complete('Export failed: ' + error.message);
        }
    } finally {
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.style.opacity = '';
            sendButton.style.pointerEvents = '';
        }
        isSubmitting = false;
        promptInput.focus();
        
        // Auto-manage chat history after new messages
        autoManageChatHistory();
    }
}

// Fallback basic form submission function
async function basicFormSubmission(userInput, project, voices, random) {
    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_input: userInput,
                project: project,
                voices: voices || ['SyntaxPrime'],
                random: random || false
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Simple response handling if streaming isn't available
        const data = await response.json();
        
        // Add bot response
        if (data.responses) {
            Object.entries(data.responses).forEach(([voice, content]) => {
                addBotMessage(voice, content);
            });
        }
    } catch (error) {
        console.error('Basic form submission failed:', error);
        throw error;
    }
}

// Helper functions for form submission
function getSelectedVoices() {
    const checkboxes = document.querySelectorAll('input[name="voices"]:checked');
    const voices = Array.from(checkboxes).map(cb => cb.value);
    return voices.length > 0 ? voices : ['SyntaxPrime'];
}

function addUserMessage(content) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) {
        console.error('Thread elements not found');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-bubble user">
            <div class="message-header">You</div>
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

function addBotMessage(voice, content) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) {
        console.error('Thread elements not found');
        return;
    }
    
    const displayNames = {
        'SyntaxPrime': 'Syntax Prime',
        'SyntaxBot': 'SyntaxBot',
        'Nil.exe': 'Nil.exe',
        'GGPT': 'GGPT'
    };
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="message-bubble bot">
            <div class="message-header">
                <img src="/static/syntax-buffering.png" alt="${displayNames[voice] || voice}" class="logo">
                ${displayNames[voice] || voice}
            </div>
            <div class="message-content">${renderMarkdown(content)}</div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

function addErrorMessage(content) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) {
        console.error('Thread elements not found');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="message-bubble bot" style="border-color: #ef4444;">
            <div class="message-header">
                <img src="/static/syntax-buffering.png" alt="System" class="logo">
                System Error
            </div>
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// =============================================================================
// STREAMING CHAT AND MARKDOWN RENDERING
// =============================================================================

function startStreamingChat(userInput, project, voices, random, options = {}) {
    return new Promise((resolve, reject) => {
        fetch('/api/chat/stream', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_input: userInput,
                project: project,
                voices: voices,
                random: random
            })
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            const responseData = {};
            
            function readStream() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        resolve(responseData);
                        return;
                    }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    
                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                switch (data.type) {
                                    case 'start':
                                        showStreamingStatus(data.message);
                                        break;
                                        
                                    case 'content':
                                        if (!responseData[data.voice]) {
                                            responseData[data.voice] = '';
                                            addStreamingMessage(data.voice);
                                        }
                                        responseData[data.voice] += data.chunk;
                                        updateStreamingMessage(data.voice, responseData[data.voice]);
                                        break;
                                        
                                    case 'image':
                                        handleInlineImageData(data.image_data, data.image_url);
                                        break;
                                        
                                    case 'complete':
                                        hideStreamingStatus();
                                        finalizeStreamingMessages(data.responses);
                                        resolve(data.responses);
                                        break;
                                        
                                    case 'error':
                                        hideStreamingStatus();
                                        reject(new Error(data.message));
                                        break;
                                }
                            } catch (e) {
                                console.error('Failed to parse SSE data:', e);
                            }
                        }
                    });
                    
                    return readStream();
                });
            }
            
            return readStream();
            
        }).catch(error => {
            hideStreamingStatus();
            console.error('Streaming fetch failed:', error);
            reject(error);
        });
    });
}

function handleInlineImageData(imageData, imageUrl) {
    const botMessages = document.querySelectorAll('.message.bot');
    const lastBotMessage = botMessages[botMessages.length - 1];
    
    if (lastBotMessage && imageData && imageUrl) {
        const responseDataStr = JSON.stringify({
            image_data: imageData,
            image_url: imageUrl
        });
        
        lastBotMessage.setAttribute('data-response-data', responseDataStr);
        imageHandler.handleResponseData(lastBotMessage, responseDataStr);
    }
}

function showStreamingStatus(message) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    const statusDiv = document.createElement('div');
    statusDiv.className = 'streaming-status';
    statusDiv.id = 'streamingStatus';
    statusDiv.innerHTML = `
        <div class="streaming-indicator"></div>
        <span>${message}</span>
    `;
    
    thread.insertBefore(statusDiv, anchor);
    scrollToBottom();
}

function hideStreamingStatus() {
    const status = document.getElementById('streamingStatus');
    if (status) {
        status.remove();
    }
}

function addStreamingMessage(voice) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    const displayNames = {
        'SyntaxPrime': 'Syntax Prime',
        'SyntaxBot': 'SyntaxBot',
        'Nil.exe': 'Nil.exe',
        'GGPT': 'GGPT'
    };
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="message-bubble bot">
            <div class="message-header">
                <img src="/static/syntax-buffering.png" alt="${displayNames[voice] || voice}" class="logo">
                ${displayNames[voice] || voice}
                <button class="speaker-btn" onclick="speakText(this.closest('.message-bubble').querySelector('.message-content').textContent, this)" title="Speak this message">
                    🔊
                </button>
            </div>
            <div class="message-content" id="streaming-${voice}"></div>
            <button class="mobile-copy-btn" onclick="copyToClipboard(this.closest('.message-bubble').querySelector('.message-content').textContent, this)" title="Copy this response">
                📋 Copy
            </button>
            <div class="feedback-buttons">
                <button class="feedback-btn thumbs-up" onclick="recordFeedback('${Date.now()}_${voice}', 'thumbs_up', this)" title="Good response">
                    👍
                </button>
                <button class="feedback-btn thumbs-down" onclick="recordFeedback('${Date.now()}_${voice}', 'thumbs_down', this)" title="Poor response">
                    👎
                </button>
                <button class="feedback-btn middle-finger" onclick="recordFeedback('${Date.now()}_${voice}', 'middle_finger', this)" title="Sarcastic/unhelpful response">
                    🖕
                </button>
            </div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

// Enhanced markdown rendering function
function renderMarkdown(text) {
    if (!text) return '';
    
    return text
        // Headers (must be done first)
        .replace(/^### (.*$)/gm, '<h3 class="chat-h3">$1</h3>')
        .replace(/^## (.*$)/gm, '<h2 class="chat-h2">$1</h2>')
        .replace(/^# (.*$)/gm, '<h1 class="chat-h1">$1</h1>')
        
        // Bold and italic (improved to avoid conflicts)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>')
        
        // Code blocks and inline code
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`\n]+)`/g, '<code>$1</code>')
        
        // Lists - improved handling
        .replace(/^\* (.+)$/gm, '<li>$1</li>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/((?:<li>.*<\/li>\s*\n?)+)/gm, '<ul>$1</ul>')
        
        // Numbered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        .replace(/((?:<li>.*<\/li>\s*\n?)+)(?![^<]*<ul>)/gm, '<ol>$1</ol>')
        
        // Blockquotes
        .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
        
        // Line breaks - preserve double breaks as paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        
        // Wrap in paragraphs (avoid wrapping headers, lists, blockquotes, code blocks)
        .replace(/^(?!<[h1-6|ul|ol|blockquote|pre])(.*?)$/gm, '<p>$1</p>')
        
        // Clean up empty paragraphs and nested tags
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<h[1-6][^>]*>.*?<\/h[1-6]>)<\/p>/g, '$1')
        .replace(/<p>(<ul>.*?<\/ul>)<\/p>/gs, '$1')
        .replace(/<p>(<ol>.*?<\/ol>)<\/p>/gs, '$1')
        .replace(/<p>(<blockquote>.*?<\/blockquote>)<\/p>/g, '$1')
        .replace(/<p>(<pre>.*?<\/pre>)<\/p>/gs, '$1')
        
        // Fix broken paragraph tags
        .replace(/<p><br>/g, '<p>')
        .replace(/<br><\/p>/g, '</p>');
}

function updateStreamingMessage(voice, content) {
    const element = document.getElementById(`streaming-${voice}`);
    if (element) {
        element.innerHTML = renderMarkdown(content);
        scrollToBottom();
    }
}

function finalizeStreamingMessages(responseData) {
    const streamingElements = document.querySelectorAll('[id^="streaming-"]');
    streamingElements.forEach(el => {
        el.removeAttribute('id');
    });
    
    // Check if any response contains image data and add it to the last message
    if (responseData) {
        for (const [voice, response] of Object.entries(responseData)) {
            if (typeof response === 'object' && response.image_data && response.image_url) {
                const botMessages = document.querySelectorAll('.message.bot');
                const lastBotMessage = botMessages[botMessages.length - 1];
                
                if (lastBotMessage) {
                    const responseDataStr = JSON.stringify(response);
                    lastBotMessage.setAttribute('data-response-data', responseDataStr);
                    imageHandler.handleResponseData(lastBotMessage, responseDataStr);
                }
            }
        }
    }
    
    // Auto-scroll after finalization
    setTimeout(scrollToBottom, 100);
}

// =============================================================================
// FIXED FORM HANDLING AND APP INITIALIZATION
// =============================================================================

function initializeApp() {
    console.log('Initializing app...');
    
    try {
        // Initialize timezone detection
        initializeTimezoneDetection();
        
        // Initialize menu system
        initializeMenu();
        
        // Initialize file handling
        initializeFileHandling();
        
        const promptInput = document.getElementById('promptInput');
        const sendButton = document.getElementById('sendBtn');

        if (promptInput) {
            console.log('Setting up prompt input handlers');
            
            // Remove any existing event listeners by cloning the element
            const newPromptInput = promptInput.cloneNode(true);
            promptInput.parentNode.replaceChild(newPromptInput, promptInput);
            
            // Add fresh event listeners
            newPromptInput.addEventListener('input', () => autoResize(newPromptInput));
            newPromptInput.addEventListener('keydown', handleKeyDown);
            newPromptInput.addEventListener('focus', handleInputFocus);
            newPromptInput.addEventListener('blur', handleInputBlur);
            
            autoResize(newPromptInput);
            console.log('Prompt input handlers set up successfully');
        } else {
            console.error('Prompt input not found!');
        }

        if (sendButton) {
            console.log('Setting up send button handlers');
            
            // Remove any existing event listeners by cloning the element
            const newSendButton = sendButton.cloneNode(true);
            sendButton.parentNode.replaceChild(newSendButton, sendButton);
            
            // Add fresh event listeners
            newSendButton.addEventListener('click', handleSendClick);
            newSendButton.addEventListener('touchstart', (e) => {
                e.preventDefault();
            });
            
            newSendButton.addEventListener('contextmenu', (e) => {
                e.preventDefault();
            });
            
            console.log('Send button handlers set up successfully');
        } else {
            console.error('Send button not found!');
        }

        const projectSelect = document.getElementById('projectSelect');
        if (projectSelect) {
            projectSelect.addEventListener('change', updateExportLink);
        }
        
        // Initialize with recent-only view if we have many messages
        const messages = document.querySelectorAll('.message');
        if (messages.length > chatHistoryManager.maxVisibleMessages) {
            chatHistoryManager.showingRecentOnly = true;
            toggleHistoryView();
        }
        
        console.log('App initialized successfully');
    } catch (error) {
        console.error('Failed to initialize app:', error);
    }
}

function autoResize(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    const maxHeight = 120;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = newHeight + 'px';
}

function handleKeyDown(e) {
    // Only submit on Enter for desktop/laptop (not mobile)
    const isMobile = window.innerWidth <= 768 || /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (e.key === 'Enter' && !e.shiftKey && !isMobile) {
        e.preventDefault();
        submitForm();
    }
}

function handleInputFocus() {
    setTimeout(() => {
        document.getElementById('promptInput')?.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }, 300);
}

function handleInputBlur() {
    // Only scroll to top on mobile to prevent keyboard issues
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
        setTimeout(() => {
            window.scrollTo(0, 0);
        }, 100);
    }
}

function handleSendClick(e) {
    console.log('Send button clicked');
    e.preventDefault();
    e.stopPropagation();
    submitForm();
}

function scrollToBottom() {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
    } else if (thread) {
        thread.scrollTop = thread.scrollHeight;
    }
}

function updateExportLink() {
    const projectSelect = document.getElementById('projectSelect');
    const exportLink = document.querySelector('a[href*="/export/"]');
    
    if (projectSelect && exportLink) {
        const selectedProject = encodeURIComponent(projectSelect.value);
        exportLink.href = `/export/${selectedProject}`;
    }
}

// =============================================================================
// FILE UPLOAD AND ANALYSIS
// =============================================================================

async function handleFileUploadAndAnalysis(userInput, project) {
    if (!attachedFiles || attachedFiles.length === 0) {
        throw new Error('No files to upload');
    }
    
    const formData = new FormData();
    formData.append('user_input', userInput);
    formData.append('project', project);
    
    // Add all attached files
    attachedFiles.forEach((file, index) => {
        formData.append(`files`, file);
    });
    
    try {
        const response = await fetch('/api/chat/upload', {
            method: 'POST',
            credentials: 'include',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Add bot responses to chat
        if (result.responses) {
            Object.entries(result.responses).forEach(([voice, content]) => {
                addBotMessage(voice, content);
            });
        }
        
        // Clear attached files after successful upload
        attachedFiles = [];
        updateAttachedFilesUI();
        updatePaperclipButton();
        
        return result;
        
    } catch (error) {
        console.error('File upload failed:', error);
        throw error;
    }
}

// =============================================================================
// GLOBAL FUNCTION EXPOSURE
// =============================================================================

// Make functions globally accessible for onclick handlers and external calls
window.submitForm = submitForm;
window.handleKeyDown = handleKeyDown;
window.handleSendClick = handleSendClick;
window.sendQuickCommand = sendQuickCommand;
window.refreshPage = refreshPage;
window.copyToClipboard = copyToClipboard;
window.speakText = speakText;
window.recordFeedback = recordFeedback;
window.removeAttachedFile = removeAttachedFile;
window.toggleHistoryView = toggleHistoryView;
window.clearOldMessages = clearOldMessages;
window.imageHandler = imageHandler;

// Global function to ensure menu items work
window.executeMenuAction = function(action, param) {
    console.log('Executing menu action:', action, param);
    
    switch(action) {
        case 'quickCommand':
            sendQuickCommand(param);
            break;
        case 'navigate':
            window.location.href = param;
            break;
        case 'reload':
            refreshPage();
            break;
        default:
            console.warn('Unknown menu action:', action);
    }
    
    closeMenu();
};

// =============================================================================
// INITIALIZATION
// =============================================================================

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Starting initialization');
    
    // Initialize thread sidebar
    threadSidebar = new ThreadSidebar();
    window.threadSidebar = threadSidebar; // Make globally accessible
    
    // Initialize main app
    initializeApp();
    
    // Test that elements exist
    const promptInput = document.getElementById('promptInput');
    const sendButton = document.getElementById('sendBtn');
    
    console.log('Elements found:', {
        promptInput: !!promptInput,
        sendButton: !!sendButton
    });
});

// Also initialize on window load for safety
window.addEventListener('load', function() {
    console.log('Window loaded - Running final setup');
    setTimeout(() => {
        scrollToBottom();
        
        if (!window.location.hash) {
            const promptInput = document.getElementById('promptInput');
            if (promptInput) {
                promptInput.focus();
            }
        }
    }, 100);
    
    if (window.location.hash === '#bottom-anchor') {
        setTimeout(scrollToBottom, 500);
    }
});

console.log('Complete Ghostline app.js loaded successfully');
