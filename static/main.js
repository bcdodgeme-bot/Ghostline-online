document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const outputArea = document.querySelector("main");
    const loader = document.getElementById("loader");

    form.addEventListener("submit", function () {
        if (loader) loader.style.display = "block";
    });

    if (outputArea) {
        outputArea.scrollTop = outputArea.scrollHeight;
    }

    // Initialize search functionality
    initializeSearch();
});

// ========================================
// SEARCH FUNCTIONALITY
// ========================================

let searchState = {
    isOpen: false,
    currentQuery: '',
    currentFilters: {
        project: 'all',
        dateRange: 'all',
        searchType: 'all'
    },
    results: [],
    selectedIndex: -1
};

function initializeSearch() {
    // Create search interface
    createSearchInterface();
    
    // Bind keyboard shortcuts
    document.addEventListener('keydown', handleSearchShortcuts);
    
    // Initialize search input handlers
    setupSearchHandlers();
    
    console.log('Search functionality initialized');
}

function createSearchInterface() {
    const searchHTML = `
        <!-- Search Overlay -->
        <div id="searchOverlay" class="search-overlay">
            <div class="search-container">
                <div class="search-header">
                    <div class="search-input-container">
                        <input type="text" id="searchInput" placeholder="Search conversations..." autocomplete="off">
                        <button id="searchClose" class="search-close">×</button>
                    </div>
                </div>
                
                <div class="search-filters">
                    <select id="projectFilter">
                        <option value="all">All Projects</option>
                        <option value="Personal Operating Manual">Personal Operating Manual</option>
                        <option value="AMCF">AMCF</option>
                        <option value="BCDodgeme">BCDodgeme</option>
                        <option value="Rose and Angel">Rose and Angel</option>
                        <option value="Meals N Feelz">Meals N Feelz</option>
                        <option value="TV Signals">TV Signals</option>
                        <option value="Damn It Carl">Damn It Carl</option>
                        <option value="HalalBot">HalalBot</option>
                        <option value="Kitchen">Kitchen</option>
                        <option value="Health">Health</option>
                        <option value="Side Quests">Side Quests</option>
                    </select>
                    
                    <select id="dateFilter">
                        <option value="all">All Time</option>
                        <option value="today">Today</option>
                        <option value="week">This Week</option>
                        <option value="month">This Month</option>
                        <option value="3months">Last 3 Months</option>
                    </select>
                    
                    <select id="typeFilter">
                        <option value="all">All Types</option>
                        <option value="conversations">Conversations</option>
                        <option value="threads">Threads</option>
                        <option value="bookmarks">Bookmarks</option>
                    </select>
                    
                    <div class="search-options">
                        <label>
                            <input type="checkbox" id="includeContext" checked>
                            Include AI responses
                        </label>
                    </div>
                </div>
                
                <div class="search-results-container">
                    <div id="searchResults" class="search-results">
                        <div class="search-placeholder">
                            Start typing to search conversations...
                            <div class="search-tips">
                                <div><kbd>Ctrl+K</kbd> to open search</div>
                                <div><kbd>↑↓</kbd> to navigate results</div>
                                <div><kbd>Enter</kbd> to select</div>
                                <div><kbd>Esc</kbd> to close</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- In-thread Search Bar -->
        <div id="threadSearchBar" class="thread-search-bar hidden">
            <input type="text" id="threadSearchInput" placeholder="Search in current conversation...">
            <button id="threadSearchPrev" class="thread-search-btn">↑</button>
            <button id="threadSearchNext" class="thread-search-btn">↓</button>
            <span id="threadSearchCounter" class="thread-search-counter">0/0</span>
            <button id="threadSearchClose" class="thread-search-close">×</button>
        </div>
    `;
    
    // Add search styles
    const searchStyles = `
        <style>
        .search-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(4px);
            z-index: 1000;
            display: none;
            align-items: flex-start;
            justify-content: center;
            padding-top: 10vh;
        }
        
        .search-overlay.show {
            display: flex;
        }
        
        .search-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-lg);
        }
        
        .search-header {
            padding: 20px 20px 0 20px;
        }
        
        .search-input-container {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        .search-input-container input {
            width: 100%;
            padding: 16px 50px 16px 20px;
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            color: var(--text-primary);
            font-size: 18px;
            outline: none;
        }
        
        .search-input-container input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .search-close {
            position: absolute;
            right: 15px;
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 24px;
            cursor: pointer;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }
        
        .search-close:hover {
            background: var(--surface-hover);
            color: var(--text-primary);
        }
        
        .search-filters {
            padding: 15px 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        
        .search-filters select {
            background: var(--background);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 8px 12px;
            font-size: 14px;
        }
        
        .search-options {
            margin-left: auto;
        }
        
        .search-options label {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-secondary);
            font-size: 14px;
            cursor: pointer;
        }
        
        .search-results-container {
            flex: 1;
            overflow: hidden;
        }
        
        .search-results {
            height: 100%;
            overflow-y: auto;
            max-height: 400px;
        }
        
        .search-results::-webkit-scrollbar {
            width: 6px;
        }
        
        .search-results::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        .search-placeholder {
            padding: 40px 20px;
            text-align: center;
            color: var(--text-muted);
        }
        
        .search-tips {
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px;
            font-size: 12px;
        }
        
        .search-tips kbd {
            background: var(--surface-hover);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
        
        .search-result {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .search-result:hover,
        .search-result.selected {
            background: var(--surface-hover);
        }
        
        .search-result.selected {
            border-left: 3px solid var(--primary);
        }
        
        .search-result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .search-result-type {
            background: var(--primary);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        
        .search-result-type.bookmark {
            background: var(--accent);
        }
        
        .search-result-type.thread {
            background: var(--success);
        }
        
        .search-result-date {
            color: var(--text-muted);
            font-size: 12px;
        }
        
        .search-result-title {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 6px;
            line-height: 1.3;
        }
        
        .search-result-content {
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        .search-result-project {
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 6px;
        }
        
        .search-loading {
            padding: 30px;
            text-align: center;
            color: var(--primary);
        }
        
        .search-no-results {
            padding: 40px 20px;
            text-align: center;
            color: var(--text-muted);
        }
        
        .search-highlight {
            background: rgba(255, 255, 0, 0.3);
            color: var(--text-primary);
            font-weight: 500;
            padding: 1px 2px;
            border-radius: 2px;
        }
        
        /* Thread Search Bar */
        .thread-search-bar {
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 100;
            transition: transform 0.3s ease;
        }
        
        .thread-search-bar.hidden {
            transform: translateX(120%);
        }
        
        .thread-search-bar input {
            width: 200px;
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 6px 10px;
            color: var(--text-primary);
            font-size: 14px;
            outline: none;
        }
        
        .thread-search-btn,
        .thread-search-close {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 4px 8px;
            border-radius: var(--radius-sm);
            font-size: 14px;
        }
        
        .thread-search-btn:hover,
        .thread-search-close:hover {
            background: var(--surface-hover);
            color: var(--text-primary);
        }
        
        .thread-search-counter {
            color: var(--text-muted);
            font-size: 12px;
            min-width: 40px;
        }
        
        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            .search-container {
                width: 95%;
                max-height: 90vh;
                margin-top: 5vh;
            }
            
            .search-filters {
                flex-direction: column;
                align-items: stretch;
            }
            
            .search-filters > * {
                margin-bottom: 8px;
            }
            
            .thread-search-bar {
                position: relative;
                top: auto;
                right: auto;
                margin: 10px;
                width: calc(100% - 20px);
            }
            
            .thread-search-bar input {
                flex: 1;
                width: auto;
            }
        }
        </style>
    `;
    
    // Add to document
    document.head.insertAdjacentHTML('beforeend', searchStyles);
    document.body.insertAdjacentHTML('beforeend', searchHTML);
}

function setupSearchHandlers() {
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const projectFilter = document.getElementById('projectFilter');
    const dateFilter = document.getElementById('dateFilter');
    const typeFilter = document.getElementById('typeFilter');
    const includeContext = document.getElementById('includeContext');
    const searchClose = document.getElementById('searchClose');
    const threadSearchInput = document.getElementById('threadSearchInput');
    const threadSearchClose = document.getElementById('threadSearchClose');
    const threadSearchPrev = document.getElementById('threadSearchPrev');
    const threadSearchNext = document.getElementById('threadSearchNext');
    
    // Debounced search
    let searchTimeout;
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim();
        
        if (query.length === 0) {
            showSearchPlaceholder();
            return;
        }
        
        searchTimeout = setTimeout(() => {
            performSearch(query);
        }, 300);
    });
    
    // Filter changes trigger new search
    [projectFilter, dateFilter, typeFilter, includeContext].forEach(filter => {
        filter.addEventListener('change', () => {
            const query = searchInput.value.trim();
            if (query) {
                performSearch(query);
            }
        });
    });
    
    // Navigation
    searchInput.addEventListener('keydown', handleSearchNavigation);
    
    // Close handlers
    searchClose.addEventListener('click', closeSearch);
    document.getElementById('searchOverlay').addEventListener('click', function(e) {
        if (e.target === this) {
            closeSearch();
        }
    });
    
    // Thread search handlers
    threadSearchInput.addEventListener('input', debounce(performThreadSearch, 200));
    threadSearchClose.addEventListener('click', closeThreadSearch);
    threadSearchPrev.addEventListener('click', () => navigateThreadSearch(-1));
    threadSearchNext.addEventListener('click', () => navigateThreadSearch(1));
    
    console.log('Search handlers initialized');
}

function handleSearchShortcuts(e) {
    // Ctrl+K or Cmd+K to open search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        openSearch();
        return;
    }
    
    // Ctrl+F or Cmd+F to open thread search
    if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !searchState.isOpen) {
        e.preventDefault();
        openThreadSearch();
        return;
    }
    
    // Escape to close any search
    if (e.key === 'Escape') {
        if (searchState.isOpen) {
            closeSearch();
        } else if (!document.getElementById('threadSearchBar').classList.contains('hidden')) {
            closeThreadSearch();
        }
    }
}

function openSearch() {
    searchState.isOpen = true;
    const overlay = document.getElementById('searchOverlay');
    const input = document.getElementById('searchInput');
    
    overlay.classList.add('show');
    
    // Focus with delay to ensure proper display
    setTimeout(() => {
        input.focus();
        input.select();
    }, 100);
    
    showSearchPlaceholder();
}

function closeSearch() {
    searchState.isOpen = false;
    const overlay = document.getElementById('searchOverlay');
    
    overlay.classList.remove('show');
    
    // Clear search state
    searchState.results = [];
    searchState.selectedIndex = -1;
    document.getElementById('searchInput').value = '';
}

function openThreadSearch() {
    const bar = document.getElementById('threadSearchBar');
    const input = document.getElementById('threadSearchInput');
    
    bar.classList.remove('hidden');
    setTimeout(() => {
        input.focus();
        input.select();
    }, 100);
}

function closeThreadSearch() {
    const bar = document.getElementById('threadSearchBar');
    
    bar.classList.add('hidden');
    clearThreadHighlights();
    updateThreadSearchCounter(0, 0);
}

function performSearch(query) {
    if (!query.trim()) {
        showSearchPlaceholder();
        return;
    }
    
    searchState.currentQuery = query;
    
    // Update filters
    searchState.currentFilters = {
        project: document.getElementById('projectFilter').value,
        dateRange: document.getElementById('dateFilter').value,
        searchType: document.getElementById('typeFilter').value
    };
    
    const includeContext = document.getElementById('includeContext').checked;
    
    showSearchLoading();
    
    // Build search request
    const searchData = {
        query: query,
        project: searchState.currentFilters.project === 'all' ? null : searchState.currentFilters.project,
        limit: 50,
        search_type: searchState.currentFilters.searchType,
        include_context: includeContext
    };
    
    // Apply date filtering (this could be enhanced with server-side support)
    if (searchState.currentFilters.dateRange !== 'all') {
        searchData.date_filter = searchState.currentFilters.dateRange;
    }
    
    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(searchData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Search failed: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displaySearchResults(data.search_results, query);
        } else {
            showSearchError(data.error || 'Search failed');
        }
    })
    .catch(error => {
        console.error('Search error:', error);
        showSearchError('Search request failed');
    });
}

function displaySearchResults(results, query) {
    const container = document.getElementById('searchResults');
    
    // Combine all result types
    const allResults = [
        ...(results.conversations || []),
        ...(results.threads || []),
        ...(results.bookmarks || [])
    ];
    
    if (allResults.length === 0) {
        container.innerHTML = `
            <div class="search-no-results">
                No results found for "${escapeHtml(query)}"
                <div style="margin-top: 10px; font-size: 14px; color: var(--text-muted);">
                    Try different keywords or adjust filters
                </div>
            </div>
        `;
        return;
    }
    
    // Sort by relevance/date
    allResults.sort((a, b) => {
        if (a.relevance_score && b.relevance_score) {
            return b.relevance_score - a.relevance_score;
        }
        return new Date(b.created_at) - new Date(a.created_at);
    });
    
    searchState.results = allResults;
    searchState.selectedIndex = 0;
    
    const resultsHTML = allResults.map((result, index) =>
        renderSearchResult(result, query, index)
    ).join('');
    
    container.innerHTML = resultsHTML;
    
    // Add click handlers
    container.querySelectorAll('.search-result').forEach((element, index) => {
        element.addEventListener('click', () => selectSearchResult(index));
    });
    
    // Highlight first result
    updateSelectedResult();
}

function renderSearchResult(result, query, index) {
    const date = new Date(result.created_at).toLocaleDateString();
    const time = new Date(result.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let title, content, typeClass;
    
    switch(result.type) {
        case 'conversation':
            title = highlightText(truncateText(result.user_input, 80), query);
            content = result.ai_response_preview ?
                highlightText(truncateText(result.ai_response_preview, 120), query) :
                highlightText(truncateText(result.user_input, 120), query);
            typeClass = 'conversation';
            break;
            
        case 'thread':
            title = highlightText(result.title || 'Untitled Thread', query);
            content = `${result.message_count} messages`;
            typeClass = 'thread';
            break;
            
        case 'bookmark':
            title = highlightText(result.title || 'Untitled Bookmark', query);
            content = result.notes ?
                highlightText(truncateText(result.notes, 100), query) :
                (result.conversation_preview ? highlightText(truncateText(result.conversation_preview, 100), query) : '');
            typeClass = 'bookmark';
            break;
            
        default:
            title = 'Unknown Result';
            content = '';
            typeClass = 'unknown';
    }
    
    return `
        <div class="search-result" data-index="${index}">
            <div class="search-result-header">
                <span class="search-result-type ${typeClass}">${result.type}</span>
                <span class="search-result-date">${date} ${time}</span>
            </div>
            <div class="search-result-title">${title}</div>
            ${content ? `<div class="search-result-content">${content}</div>` : ''}
            <div class="search-result-project">📁 ${result.project || 'No Project'}</div>
        </div>
    `;
}

function selectSearchResult(index) {
    const result = searchState.results[index];
    if (!result) return;
    
    // Close search
    closeSearch();
    
    // Navigate to result based on type
    switch(result.type) {
        case 'conversation':
            scrollToConversation(result.id);
            break;
            
        case 'thread':
            if (result.thread_id) {
                loadThread(result.thread_id);
            }
            break;
            
        case 'bookmark':
            if (result.chat_id) {
                scrollToConversation(result.chat_id);
            } else if (result.thread_id) {
                loadThread(result.thread_id);
            }
            break;
    }
}

function scrollToConversation(conversationId) {
    // Find the conversation element in the current view
    const element = document.querySelector(`[data-conversation-id="${conversationId}"]`);
    
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
        
        // Highlight briefly
        element.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
        element.style.transition = 'background-color 0.3s ease';
        
        setTimeout(() => {
            element.style.backgroundColor = '';
        }, 2000);
    } else {
        // If not found in current view, we might need to load that conversation
        // This would require additional backend support
        console.log('Conversation not found in current view:', conversationId);
    }
}

function loadThread(threadId) {
    // Navigate to thread view - this would depend on your routing system
    // For now, just log it
    console.log('Load thread:', threadId);
    
    // You might want to implement this based on your app's navigation
    // window.location.href = `/thread/${threadId}`;
}

function handleSearchNavigation(e) {
    if (!searchState.results.length) return;
    
    switch(e.key) {
        case 'ArrowDown':
            e.preventDefault();
            searchState.selectedIndex = Math.min(
                searchState.selectedIndex + 1,
                searchState.results.length - 1
            );
            updateSelectedResult();
            break;
            
        case 'ArrowUp':
            e.preventDefault();
            searchState.selectedIndex = Math.max(searchState.selectedIndex - 1, 0);
            updateSelectedResult();
            break;
            
        case 'Enter':
            e.preventDefault();
            if (searchState.selectedIndex >= 0) {
                selectSearchResult(searchState.selectedIndex);
            }
            break;
    }
}

function updateSelectedResult() {
    const results = document.querySelectorAll('.search-result');
    
    results.forEach((result, index) => {
        if (index === searchState.selectedIndex) {
            result.classList.add('selected');
            result.scrollIntoView({ block: 'nearest' });
        } else {
            result.classList.remove('selected');
        }
    });
}

// ========================================
// THREAD SEARCH FUNCTIONALITY
// ========================================

let threadSearchState = {
    query: '',
    matches: [],
    currentIndex: -1
};

function performThreadSearch() {
    const query = document.getElementById('threadSearchInput').value.trim();
    
    if (!query) {
        clearThreadHighlights();
        updateThreadSearchCounter(0, 0);
        threadSearchState = { query: '', matches: [], currentIndex: -1 };
        return;
    }
    
    threadSearchState.query = query;
    
    // Clear previous highlights
    clearThreadHighlights();
    
    // Find all text nodes in the chat thread
    const thread = document.getElementById('thread') || document.querySelector('.thread');
    if (!thread) {
        updateThreadSearchCounter(0, 0);
        return;
    }
    
    // Find matches
    const matches = findTextMatches(thread, query);
    threadSearchState.matches = matches;
    threadSearchState.currentIndex = matches.length > 0 ? 0 : -1;
    
    // Highlight matches
    highlightMatches(matches);
    
    // Update counter
    updateThreadSearchCounter(threadSearchState.currentIndex + 1, matches.length);
    
    // Scroll to first match
    if (matches.length > 0) {
        scrollToMatch(0);
    }
}

function findTextMatches(container, query) {
    const matches = [];
    const walker = document.createTreeWalker(
        container,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: function(node) {
                // Skip script and style elements
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                
                const tagName = parent.tagName.toLowerCase();
                if (['script', 'style', 'noscript'].includes(tagName)) {
                    return NodeFilter.FILTER_REJECT;
                }
                
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );
    
    let node;
    while (node = walker.nextNode()) {
        const text = node.textContent;
        const regex = new RegExp(escapeRegExp(query), 'gi');
        let match;
        
        while ((match = regex.exec(text)) !== null) {
            matches.push({
                node: node,
                start: match.index,
                end: match.index + match[0].length,
                text: match[0]
            });
        }
    }
    
    return matches;
}

function highlightMatches(matches) {
    // Process matches in reverse order to maintain text offsets
    for (let i = matches.length - 1; i >= 0; i--) {
        const match = matches[i];
        const node = match.node;
        const start = match.start;
        const end = match.end;
        
        // Create highlight element
        const highlight = document.createElement('span');
        highlight.className = 'thread-search-highlight';
        highlight.setAttribute('data-match-index', i);
        
        // Split text node
        const beforeText = node.textContent.substring(0, start);
        const matchText = node.textContent.substring(start, end);
        const afterText = node.textContent.substring(end);
        
        // Replace original node with highlighted version
        const parent = node.parentNode;
        
        if (beforeText) {
            parent.insertBefore(document.createTextNode(beforeText), node);
        }
        
        highlight.textContent = matchText;
        parent.insertBefore(highlight, node);
        
        if (afterText) {
            parent.insertBefore(document.createTextNode(afterText), node);
        }
        
        parent.removeChild(node);
    }
    
    // Add styles for highlights
    if (!document.getElementById('threadSearchStyles')) {
        const styles = document.createElement('style');
        styles.id = 'threadSearchStyles';
        styles.textContent = `
            .thread-search-highlight {
                background: rgba(255, 255, 0, 0.4);
                color: #000;
                padding: 1px 2px;
                border-radius: 2px;
                font-weight: 500;
            }
            
            .thread-search-highlight.current {
                background: rgba(255, 165, 0, 0.6);
                box-shadow: 0 0 0 2px rgba(255, 165, 0, 0.3);
            }
        `;
        document.head.appendChild(styles);
    }
}

function clearThreadHighlights() {
    const highlights = document.querySelectorAll('.thread-search-highlight');
    
    highlights.forEach(highlight => {
        const parent = highlight.parentNode;
        if (parent) {
            parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
            parent.normalize(); // Merge adjacent text nodes
        }
    });
}

function navigateThreadSearch(direction) {
    const matches = threadSearchState.matches;
    
    if (matches.length === 0) return;
    
    // Remove current highlight
    const currentHighlight = document.querySelector('.thread-search-highlight.current');
    if (currentHighlight) {
        currentHighlight.classList.remove('current');
    }
    
    // Update index
    if (direction > 0) {
        threadSearchState.currentIndex = (threadSearchState.currentIndex + 1) % matches.length;
    } else {
        threadSearchState.currentIndex =
            threadSearchState.currentIndex <= 0 ? matches.length - 1 : threadSearchState.currentIndex - 1;
    }
    
    // Update counter and scroll
    updateThreadSearchCounter(threadSearchState.currentIndex + 1, matches.length);
    scrollToMatch(threadSearchState.currentIndex);
}

function scrollToMatch(index) {
    const highlight = document.querySelector(`[data-match-index="${index}"]`);
    
    if (highlight) {
        // Remove previous current class
        document.querySelectorAll('.thread-search-highlight.current').forEach(el => {
            el.classList.remove('current');
        });
        
        // Add current class
        highlight.classList.add('current');
        
        // Scroll into view
        highlight.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
}

function updateThreadSearchCounter(current, total) {
    const counter = document.getElementById('threadSearchCounter');
    if (counter) {
        counter.textContent = `${current}/${total}`;
    }
}

// ========================================
// UTILITY FUNCTIONS
// ========================================

function showSearchPlaceholder() {
    const container = document.getElementById('searchResults');
    container.innerHTML = `
        <div class="search-placeholder">
            Start typing to search conversations...
            <div class="search-tips">
                <div><kbd>Ctrl+K</kbd> to open search</div>
                <div><kbd>↑↓</kbd> to navigate results</div>
                <div><kbd>Enter</kbd> to select</div>
                <div><kbd>Esc</kbd> to close</div>
            </div>
        </div>
    `;
}

function showSearchLoading() {
    const container = document.getElementById('searchResults');
    container.innerHTML = '<div class="search-loading">Searching...</div>';
}

function showSearchError(error) {
    const container = document.getElementById('searchResults');
    container.innerHTML = `
        <div class="search-no-results">
            Search Error: ${escapeHtml(error)}
        </div>
    `;
}

function highlightText(text, query) {
    if (!query || !text) return text;
    
    const regex = new RegExp(`(${escapeRegExp(query)})`, 'gi');
    return text.replace(regex, '<span class="search-highlight">$1</span>');
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
