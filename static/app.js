/**
 * Ghostline Creative Intelligence System - Main Application JavaScript
 * Complete version with bookmark functionality fixes and ENHANCED IMAGE HANDLING
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
        }
    })
    .catch(error => {
        console.log('Timezone detection failed (non-critical):', error);
    });
}

// =============================================================================
// BOOKMARK SIDEBAR FUNCTIONALITY (UPDATED FOR BOOKMARKS)
// =============================================================================

class ThreadSidebar {
    constructor() {
        this.sidebar = null;
        this.overlay = null;
        this.isOpen = false;
        this.currentBookmarkId = null; // Changed from currentThreadId
        this.bookmarks = []; // Changed from threads
        this.searchTerm = '';
        
        this.init();
    }
    
    init() {
        this.createSidebarHTML();
        this.attachEventListeners();
        this.loadBookmarks(); // Changed from loadThreads
        
        // Auto-refresh bookmarks every 30 seconds
        setInterval(() => {
            if (this.isOpen) {
                this.loadBookmarks();
            }
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
                    <span>📖</span>
                    Bookmarked Conversations
                </div>
                <input type="text" class="sidebar-search" placeholder="Search bookmarks..." id="bookmarkSearch">
            </div>
            <div class="thread-list" id="bookmarkList">
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
            toggle.innerHTML = '📖'; // Changed from 📚 to 📖 to indicate bookmarks specifically
            toggle.title = 'Toggle Bookmarks';
            toggle.addEventListener('click', () => this.toggle());
            
            // Insert before the hamburger menu
            headerRight.insertBefore(toggle, headerRight.firstChild);
        }
    }
    
    attachEventListeners() {
        const searchInput = document.getElementById('bookmarkSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchTerm = e.target.value.toLowerCase();
                this.renderBookmarks();
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
        
        // Load bookmarks when opening (CHANGED: was loadThreads())
        this.loadBookmarks();
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
    
    // NEW METHOD: Load bookmarks instead of threads
    async loadBookmarks() {
        try {
            // Get current project from page context or default
            const currentProject = window.currentProject || this.getCurrentProject() || 'Personal Operating Manual';
            
            console.log(`Loading bookmarks for project: ${currentProject}`);
            
            const response = await fetch(`/api/bookmarks?project=${encodeURIComponent(currentProject)}&limit=50`, {
                credentials: 'include'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.bookmarks = data.bookmarks || [];
                console.log(`Loaded ${this.bookmarks.length} bookmarks`);
                this.renderBookmarks();
            } else {
                this.showError(`Failed to load bookmarks: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Failed to load bookmarks:', error);
            this.showError('Connection error loading bookmarks');
        }
    }
    
    // NEW METHOD: Get current project from page
    getCurrentProject() {
        // Try to get project from URL or page context
        const urlParams = new URLSearchParams(window.location.search);
        const projectFromUrl = urlParams.get('project');
        if (projectFromUrl) return projectFromUrl;
        
        // Try to get from project selector
        const projectSelect = document.getElementById('projectSelect');
        if (projectSelect && projectSelect.value) {
            return projectSelect.value;
        }
        
        // Try to get from page title or other indicators
        const titleElement = document.querySelector('title');
        if (titleElement && titleElement.textContent.includes('|')) {
            return titleElement.textContent.split('|')[1].trim();
        }
        
        // Default fallback
        return 'Personal Operating Manual';
    }
    
    // NEW METHOD: Render bookmarks instead of threads
    renderBookmarks() {
        const bookmarkList = document.getElementById('bookmarkList');
        if (!bookmarkList) return;
        
        let filteredBookmarks = this.bookmarks;
        
        // Apply search filter
        if (this.searchTerm) {
            filteredBookmarks = this.bookmarks.filter(bookmark =>
                bookmark.title.toLowerCase().includes(this.searchTerm) ||
                bookmark.preview.toLowerCase().includes(this.searchTerm) ||
                (bookmark.project && bookmark.project.toLowerCase().includes(this.searchTerm))
            );
        }
        
        if (filteredBookmarks.length === 0) {
            bookmarkList.innerHTML = this.searchTerm ?
                `<div class="thread-empty">
                    <div class="thread-empty-icon">🔍</div>
                    <div>No bookmarks match "${this.searchTerm}"</div>
                </div>` :
                `<div class="thread-empty">
                    <div class="thread-empty-icon">📖</div>
                    <div>No bookmarks created yet</div>
                    <div style="font-size: 0.8rem; margin-top: 8px; opacity: 0.7;">
                        Say "bookmark System Test" to create your first bookmark
                    </div>
                </div>`;
            return;
        }
        
        bookmarkList.innerHTML = filteredBookmarks.map(bookmark => this.renderBookmarkItem(bookmark)).join('');
        
        // Attach click listeners
        bookmarkList.querySelectorAll('.bookmark-item').forEach(item => {
            item.addEventListener('click', () => {
                const bookmarkId = item.dataset.bookmarkId;
                const chatId = item.dataset.chatId;
                this.selectBookmark(bookmarkId, chatId);
            });
        });
    }
    
    // NEW METHOD: Render individual bookmark item
    renderBookmarkItem(bookmark) {
        const isActive = bookmark.bookmark_id === this.currentBookmarkId;
        const timeAgo = this.formatTimeAgo(bookmark.created_at);
        const bookmarkType = bookmark.bookmark_type || 'manual';
        const typeIcon = bookmarkType === 'auto' ? '🤖' : bookmarkType === 'user_command' ? '👤' : '📖';
        
        return `
            <div class="bookmark-item ${isActive ? 'active' : ''}" data-bookmark-id="${bookmark.bookmark_id}" data-chat-id="${bookmark.chat_id}">
                <div class="bookmark-header">
                    <span class="bookmark-type-icon">${typeIcon}</span>
                    <span class="bookmark-title">${this.escapeHtml(bookmark.title)}</span>
                </div>
                <div class="bookmark-preview">${this.escapeHtml(bookmark.preview)}</div>
                <div class="bookmark-meta">
                    <span class="bookmark-project">${this.escapeHtml(bookmark.project)}</span>
                    <span class="bookmark-time">${timeAgo}</span>
                </div>
            </div>
        `;
    }
    
    // NEW METHOD: Select and load a bookmark
    async selectBookmark(bookmarkId, chatId) {
        if (!chatId) {
            console.error('No chatId provided for bookmark:', bookmarkId);
            return;
        }
        
        try {
            // Update UI to show selection
            const bookmarkItems = document.querySelectorAll('.bookmark-item');
            bookmarkItems.forEach(item => item.classList.remove('active'));
            
            const selectedItem = document.querySelector(`[data-bookmark-id="${bookmarkId}"]`);
            if (selectedItem) {
                selectedItem.classList.add('active');
            }
            
            this.currentBookmarkId = bookmarkId;
            
            // Load the specific conversation
            console.log(`Loading conversation ${chatId} for bookmark ${bookmarkId}`);
            
            const response = await fetch(`/api/conversation/${chatId}`, {
                credentials: 'include'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.loadConversationIntoChat(data.conversation);
                
                // Close sidebar on mobile after selection
                if (window.innerWidth <= 768) {
                    this.close();
                }
                
                // Show notification
                this.showNotification(`📖 Loaded bookmark: ${data.conversation.bookmark?.title || 'Untitled'}`);
                
                // Highlight the bookmarked message
                setTimeout(() => {
                    this.highlightBookmarkedMessage(chatId);
                }, 500);
            } else {
                this.showError(`Failed to load conversation: ${data.error}`);
            }
        } catch (error) {
            console.error('Failed to load bookmarked conversation:', error);
            this.showError('Failed to load conversation');
        }
    }
    
    // NEW METHOD: Load conversation into chat interface
    loadConversationIntoChat(conversationData) {
        const chatThread = document.getElementById('thread');
        const bottomAnchor = document.getElementById('bottom-anchor');
        
        if (!chatThread || !bottomAnchor) {
            console.error('Chat elements not found');
            return;
        }
        
        // Clear current messages (except the anchor)
        const messages = chatThread.querySelectorAll('.message');
        messages.forEach(msg => msg.remove());
        
        // Add user message
        this.addMessageToChat('user', conversationData.user_input, conversationData.bookmark);
        
        // Add AI response if exists
        if (conversationData.response_data) {
            Object.entries(conversationData.response_data).forEach(([voice, response]) => {
                if (response && typeof response === 'string') {
                    this.addMessageToChat('bot', response, conversationData.bookmark, voice);
                }
            });
        }
        
        // Scroll to bottom
        setTimeout(() => {
            bottomAnchor.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
    
    // NEW METHOD: Add message to chat
    addMessageToChat(type, content, bookmark = null, voice = null) {
        const chatThread = document.getElementById('thread');
        const bottomAnchor = document.getElementById('bottom-anchor');
        
        if (!chatThread || !bottomAnchor) return;
        
        const messageDiv = document.createElement('div');
        const isBookmarked = bookmark !== null;
        messageDiv.className = `message ${type} ${isBookmarked ? 'bookmarked' : ''}`;
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-bubble user">
                    <div class="message-header">You ${isBookmarked ? '📖' : ''}</div>
                    <div class="message-content">${this.escapeHtml(content)}</div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-bubble bot">
                    <div class="message-header">
                        <span class="logo">🤖</span>
                        ${voice || 'SyntaxPrime'} ${isBookmarked ? '📖' : ''}
                    </div>
                    <div class="message-content">${this.escapeHtml(content)}</div>
                </div>
            `;
        }
        
        chatThread.insertBefore(messageDiv, bottomAnchor);
    }
    
    // NEW METHOD: Highlight bookmarked message
    highlightBookmarkedMessage(chatId) {
        const bookmarkedMessages = document.querySelectorAll('.message.bookmarked');
        bookmarkedMessages.forEach(msg => {
            msg.classList.add('highlight-bookmark');
            setTimeout(() => {
                msg.classList.remove('highlight-bookmark');
            }, 3000);
        });
    }
    
    // Utility methods
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
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showError(message) {
        const bookmarkList = document.getElementById('bookmarkList');
        if (bookmarkList) {
            bookmarkList.innerHTML = `
                <div class="thread-empty">
                    <div class="thread-empty-icon">⚠️</div>
                    <div style="color: var(--error);">${this.escapeHtml(message)}</div>
                    <button onclick="threadSidebar.loadBookmarks()" style="margin-top: 8px; padding: 4px 8px; background: var(--surface-hover); border: 1px solid var(--border); border-radius: 4px; color: inherit; cursor: pointer;">
                        Retry
                    </button>
                </div>
            `;
        }
        console.error('Bookmark sidebar error:', message);
    }
    
    showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'bookmark-notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
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
        /^mark this$/i,
        /^remember this$/i
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
    
    // Refresh sidebar bookmarks if bookmark was successful
    if (success && window.threadSidebar) {
        setTimeout(() => {
            window.threadSidebar.loadBookmarks();
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

function addUserMessage(message) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-bubble user">
            <div class="message-header">You</div>
            <div class="message-content">${escapeHtml(message)}</div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

function addBotMessage(voice, content) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="message-bubble bot">
            <div class="message-header">
                <span class="logo">🤖</span>
                ${voice}
            </div>
            <div class="message-content">${renderMarkdown(content)}</div>
        </div>
    `;
    
    thread.insertBefore(messageDiv, anchor);
    scrollToBottom();
}

function addStreamingMessage(voice) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) return;
    
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

function addErrorMessage(message) {
    const thread = document.getElementById('thread');
    const anchor = document.getElementById('bottom-anchor');
    
    if (!thread || !anchor) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot error';
    messageDiv.innerHTML = `
        <div class="message-bubble bot">
            <div class="message-header">
                <span class="logo">⚠️</span>
                Error
            </div>
            <div class="message-content">${escapeHtml(message)}</div>
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
// ENHANCED STREAMING CHAT FUNCTIONALITY (FIXED FOR IMAGES)
// =============================================================================

async function startStreamingChat(userInput, project, voices, random, options = {}) {
    const { isBookmarkCommand, isExportCommand, exportProgress } = options;
    
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
            let pendingImageData = null; // FIXED: Store image data for later processing
            
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
                                        // FIXED: Store image data and immediately display it
                                        pendingImageData = {
                                            image_data: data.image_data,
                                            image_url: data.image_url
                                        };
                                        
                                        console.log('🖼️ Received image data for immediate display');
                                        
                                        // Immediately add to the last bot message
                                        const botMessages = document.querySelectorAll('.message.bot');
                                        const lastBotMessage = botMessages[botMessages.length - 1];
                                        
                                        if (lastBotMessage) {
                                            const responseDataStr = JSON.stringify(pendingImageData);
                                            lastBotMessage.setAttribute('data-response-data', responseDataStr);
                                            imageHandler.handleResponseData(lastBotMessage, responseDataStr);
                                        }
                                        break;
                                        
                                    case 'complete':
                                        hideStreamingStatus();
                                        
                                        // FIXED: Ensure image data is preserved in final response
                                        if (pendingImageData) {
                                            // Add image data to the response data for any voice that doesn't have content
                                            const voiceWithImage = Object.keys(data.responses)[0] || 'SyntaxPrime';
                                            if (typeof data.responses[voiceWithImage] === 'string') {
                                                // Convert string response to object with image data
                                                data.responses[voiceWithImage] = {
                                                    SyntaxPrime: data.responses[voiceWithImage],
                                                    image_data: pendingImageData.image_data,
                                                    image_url: pendingImageData.image_url
                                                };
                                            }
                                        }
                                        
                                        finalizeStreamingMessages(data.responses);
                                        
                                        if (exportProgress) {
                                            exportProgress.complete();
                                        }
                                        
                                        resolve(data.responses);
                                        break;
                                        
                                    case 'error':
                                        hideStreamingStatus();
                                        
                                        if (exportProgress) {
                                            exportProgress.complete('Export failed');
                                        }
                                        
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
            
            if (exportProgress) {
                exportProgress.complete('Streaming failed');
            }
            
            reject(error);
        });
    });
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

// =============================================================================
// MARKDOWN RENDERING
// =============================================================================

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
        .replace(/((?:<li>.*<\/li>\s*\n?)+)(?![\s\S]*<ul>)/gm, '<ol>$1</ol>')
        
        // Links
        .replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        
        // Blockquotes
        .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
        
        // Horizontal rules
        .replace(/^---$/gm, '<hr>')
        
        // Line breaks and paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/^(.+)$/gm, '<p>$1</p>')
        
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
// ENHANCED IMAGE HANDLER (FIXED)
// =============================================================================

const imageHandler = {
    handleResponseData: function(messageElement, responseDataStr) {
        try {
            const responseData = JSON.parse(responseDataStr);
            
            // Handle both direct image_data and nested image_data
            let imageData = null;
            let imageUrl = null;
            
            if (responseData.image_data && responseData.image_url) {
                imageData = responseData.image_data;
                imageUrl = responseData.image_url;
            } else if (responseData.SyntaxPrime && typeof responseData.SyntaxPrime === 'object') {
                imageData = responseData.SyntaxPrime.image_data;
                imageUrl = responseData.SyntaxPrime.image_url;
            }
            
            if (imageData && imageUrl) {
                console.log('🖼️ Processing image data for inline display');
                this.addInlineImage(messageElement, imageData, imageUrl);
            }
        } catch (error) {
            console.error('Failed to parse response data for images:', error);
        }
    },
    
    addInlineImage: function(messageElement, imageData, imageUrl) {
        const messageContent = messageElement.querySelector('.message-content');
        if (!messageContent) return;
        
        // Check if image already exists to prevent duplicates
        if (messageContent.querySelector('.inline-generated-image')) {
            console.log('Image already exists, skipping duplicate');
            return;
        }
        
        const imageContainer = document.createElement('div');
        imageContainer.style.marginTop = '15px';
        
        const img = document.createElement('img');
        
        // Handle different image data formats
        let imageSrc;
        if (typeof imageData === 'object' && imageData.data) {
            // New format: {data: base64, content_type: "image/webp"}
            const contentType = imageData.content_type || 'image/webp';
            imageSrc = `data:${contentType};base64,${imageData.data}`;
        } else if (typeof imageData === 'string') {
            // Legacy format: direct base64 string
            imageSrc = `data:image/png;base64,${imageData}`;
        } else {
            console.error('Invalid image data format:', imageData);
            return;
        }
        
        img.src = imageSrc;
        img.className = 'inline-generated-image';
        img.alt = 'Generated marketing image';
        
        // Add click handler for modal view
        img.onclick = () => this.showImageModal(imageSrc);
        
        // Add loading and error handlers
        img.onload = function() {
            console.log('✅ Image loaded successfully');
            this.style.opacity = '1';
        };
        
        img.onerror = function() {
            console.error('❌ Image failed to load');
            this.style.border = '2px solid #ef4444';
            this.alt = 'Failed to load image';
        };
        
        // Initial styling for smooth loading
        img.style.opacity = '0.5';
        img.style.transition = 'opacity 0.3s ease';
        
        const actionButtons = document.createElement('div');
        actionButtons.className = 'image-action-buttons';
        
        const downloadData = typeof imageData === 'object' ? imageData.data : imageData;
        const contentType = typeof imageData === 'object' ? imageData.content_type : 'image/png';
        
        actionButtons.innerHTML = `
            <button class="image-btn" onclick="imageHandler.downloadImage('${downloadData}', 'generated-image.png', '${contentType}')">
                💾 Download
            </button>
            <button class="image-btn secondary" onclick="imageHandler.copyImageToClipboard('${downloadData}', '${contentType}')">
                📋 Copy
            </button>
            <a href="${imageUrl}" target="_blank" class="image-btn success">
                🔗 Open Full Size
            </a>
        `;
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(actionButtons);
        messageContent.appendChild(imageContainer);
        
        // Scroll to show the new image
        setTimeout(() => {
            img.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    },
    
    showImageModal: function(src) {
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.innerHTML = `<img src="${src}" alt="Generated image" style="max-width: 90vw; max-height: 90vh;">`;
        modal.onclick = () => modal.remove();
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('show'), 10);
    },
    
    downloadImage: function(imageData, filename, contentType = 'image/png') {
        const link = document.createElement('a');
        link.href = `data:${contentType};base64,${imageData}`;
        link.download = filename;
        link.click();
    },
    
    copyImageToClipboard: function(imageData, contentType = 'image/png') {
        // Convert base64 to blob and copy to clipboard
        fetch(`data:${contentType};base64,${imageData}`)
            .then(res => res.blob())
            .then(blob => {
                const item = new ClipboardItem({[contentType]: blob});
                return navigator.clipboard.write([item]);
            })
            .then(() => {
                // Visual feedback
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✅ Copied!';
                btn.style.background = '#10b981';
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.background = '';
                }, 2000);
            })
            .catch(err => {
                console.error('Failed to copy image to clipboard:', err);
                alert('Failed to copy image to clipboard. Your browser may not support this feature.');
            });
    }
};

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
            // newSendButton.addEventListener('touchstart', (e) => {
            //    e.preventDefault();
            //});
            
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
// FEEDBACK SYSTEM
// =============================================================================

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
            
            // Get action from data attributes or onclick
            const action = menuItem.dataset.action;
            const param = menuItem.dataset.param;
            
            if (action) {
                executeMenuAction(action, param);
            }
        }
    });
    
    console.log('Menu initialized successfully');
}

function toggleMenu() {
    console.log('Toggling menu');
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
            alert(`File type not supported: ${file.name}. Supported: Images, PDF, Word documents.`);
            continue;
        }

        // Add to attached files
        attachedFiles.push(file);
    }

    // Update UI
    updateAttachedFilesUI();
    updatePaperclipButton();
    
    // Clear file input
    if (e.target.value) {
        e.target.value = '';
    }
    
    console.log(`Added ${files.length} files. Total attached: ${attachedFiles.length}`);
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
        container.innerHTML = '';
        container.style.display = 'none';
        return;
    }

    container.style.display = 'flex';
    container.innerHTML = attachedFiles.map((file, index) => `
        <div class="attached-file">
            <span class="file-icon">${getFileIcon(file)}</span>
            <span class="file-name" title="${file.name}">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button class="remove-file" onclick="removeAttachedFile(${index})" title="Remove file">×</button>
        </div>
    `).join('');
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
// UTILITY FUNCTIONS
// =============================================================================

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        console.log('Text copied to clipboard');
        if (button) {
            const originalText = button.textContent;
            button.textContent = '✅ Copied!';
            setTimeout(() => {
                button.textContent = originalText;
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy text:', err);
    });
}

function speakText(text, button) {
    if ('speechSynthesis' in window) {
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();
            if (button) button.textContent = '🔊';
            return;
        }
        
        const utterance = new SpeechSynthesisUtterance(text);
        if (button) button.textContent = '⏸️';
        
        utterance.onend = () => {
            if (button) button.textContent = '🔊';
        };
        
        speechSynthesis.speak(utterance);
    } else {
        console.warn('Speech synthesis not supported');
    }
}

// =============================================================================
// CHAT HISTORY MANAGEMENT
// =============================================================================

function toggleHistoryView() {
    const messages = document.querySelectorAll('.message');
    const toggleButton = document.getElementById('historyToggle');
    
    if (chatHistoryManager.showingRecentOnly) {
        // Show all messages
        messages.forEach(msg => {
            msg.classList.remove('fade-out', 'archived');
        });
        chatHistoryManager.showingRecentOnly = false;
        if (toggleButton) {
            toggleButton.textContent = 'Show Recent Only';
        }
    } else {
        // Show recent only
        autoManageChatHistory();
        chatHistoryManager.showingRecentOnly = true;
        const totalMessages = messages.length;
        if (toggleButton) {
            toggleButton.textContent = `Show All Messages (${totalMessages})`;
        }
    }
}

function clearOldMessages() {
    const messages = document.querySelectorAll('.message');
    const keepCount = 10;
    
    if (messages.length <= keepCount) {
        alert('No old messages to clear.');
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
    
    // Initialize thread sidebar (now with bookmark functionality)
    threadSidebar = new ThreadSidebar();
    window.threadSidebar = threadSidebar; // Make globally accessible
    
    // Initialize main app
    initializeApp();
    
    // Test that elements exist
    const promptInput = document.getElementById('promptInput');
    const sendButton = document.getElementById('sendBtn');
    
    console.log('Elements found:', {
        promptInput: !!promptInput,
        sendButton: !!sendButton,
        threadSidebar: !!threadSidebar
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

console.log('Complete Ghostline app.js with enhanced image handling loaded successfully');
