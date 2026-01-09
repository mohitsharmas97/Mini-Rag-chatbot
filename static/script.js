document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileElem');
    const processBtn = document.getElementById('process-btn');
    const uploadStatus = document.getElementById('upload-status');
    const fileList = document.getElementById('file-list');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages');
    const clearBtn = document.getElementById('clear-btn');
    const citationsPanel = document.getElementById('citations-panel');
    const citationsContent = document.getElementById('citations-content');
    const closeCitationsBtn = document.getElementById('close-citations');

    let uploadedFiles = [];

    // --- File Upload Handling ---

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files), false);

    // Also trigger file input when clicking drop area
    dropArea.addEventListener('click', () => fileInput.click());

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            uploadedFiles = Array.from(files);
            uploadStatus.textContent = `${uploadedFiles.length} file(s) selected`;
            uploadStatus.style.color = 'var(--text-primary)';
            processBtn.disabled = false;
        }
    }

    // --- Backend API Calls ---

    processBtn.addEventListener('click', async () => {
        if (uploadedFiles.length === 0) return;

        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        uploadStatus.textContent = 'Uploading and processing...';

        const formData = new FormData();
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch(`${API_CONFIG.baseURL}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                uploadStatus.textContent = 'Processing Complete!';
                uploadStatus.style.color = 'var(--success)';
                processBtn.textContent = 'Process Documents';
                processBtn.disabled = false; // Optional: allow re-upload
                updateFileList();
            } else {
                const errData = await response.json();
                throw new Error(errData.detail || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.style.color = 'var(--error)';
            processBtn.disabled = false;
            processBtn.textContent = 'Try Again';
            console.error(error);
        }
    });

    async function updateFileList() {
        try {
            const response = await fetch(`${API_CONFIG.baseURL}/api/status`);
            const data = await response.json();

            fileList.innerHTML = '';
            if (data.files && data.files.length > 0) {
                data.files.forEach(filename => {
                    const li = document.createElement('li');
                    li.textContent = filename;
                    fileList.appendChild(li);
                });
            } else {
                fileList.innerHTML = '<li class="empty-state">No files processed</li>';
            }
        } catch (e) {
            console.error("Failed to fetch status");
        }
    }

    // --- Chat Handling ---

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add User Message
        addMessage(text, 'user');
        userInput.value = '';
        userInput.style.height = 'auto'; // Reset height

        // Loading State
        const loadingId = addLoadingMessage();

        try {
            const response = await fetch(`${API_CONFIG.baseURL}/api/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: text })
            });

            // Remove loading
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) loadingMsg.remove();

            if (response.ok) {
                const data = await response.json();
                addMessage(data.answer, 'system');

                // Update Stats
                if (data.time_taken !== undefined) {
                    const statsBox = document.getElementById('stats-box');
                    const timeSpan = document.getElementById('time-taken');
                    const costSpan = document.getElementById('cost-est');

                    timeSpan.textContent = `Time: ${data.time_taken}s`;
                    costSpan.textContent = `Est. Cost: $${data.cost_estimate}`;
                    statsBox.style.display = 'flex';
                }

                if (data.citations && data.citations.length > 0) {
                    showCitations(data.citations);
                }
            } else {
                const errData = await response.json();
                addMessage(`Error: ${errData.detail || 'Something went wrong.'}`, 'system');
            }
        } catch (error) {
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) loadingMsg.remove();
            addMessage("Error: Could not connect to server.", 'system');
        }
    }

    function addMessage(text, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = type === 'user' ? '<span style="font-size: 0.8rem;">ME</span>' : '<span style="font-size: 0.8rem;">AI</span>';

        const content = document.createElement('div');
        content.className = 'content';

        // Simple markdown parsing for bold/newlines
        let formattedText = text.replace(/\n/g, '<br>');
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        content.innerHTML = formattedText;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function addLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message system`;
        msgDiv.id = id;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<span style="font-size: 0.8rem;">AI</span>';

        const content = document.createElement('div');
        content.className = 'content';
        content.innerHTML = '<em>Thinking...</em>';

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return id;
    }

    // --- Citations ---
    function showCitations(citations) {
        citationsContent.innerHTML = '';
        citations.forEach((chunk, index) => {
            const card = document.createElement('div');
            card.className = 'citation-card';
            const source = chunk.metadata ? chunk.metadata.source : 'Unknown';
            const page = chunk.metadata ? chunk.metadata.page : '?';

            card.innerHTML = `
                <div class="citation-header">
                    <strong>Citation [${index + 1}]</strong>
                    <span class="citation-meta">${source} (Page ${page})</span>
                </div>
                <p>${chunk.text.substring(0, 300)}...</p>
            `;
            citationsContent.appendChild(card);
        });
        citationsPanel.style.display = 'flex';
        setTimeout(() => citationsPanel.classList.add('active'), 10);
    }

    closeCitationsBtn.addEventListener('click', () => {
        citationsPanel.classList.remove('active');
        setTimeout(() => citationsPanel.style.display = 'none', 300);
    });

    // --- Clear & Init ---
    clearBtn.addEventListener('click', async () => {
        if (confirm("Are you sure you want to clear the knowledge base?")) {
            await fetch(`${API_CONFIG.baseURL}/api/clear`, { method: 'POST' });
            updateFileList();
            messagesContainer.innerHTML = ''; // Start fresh
            addMessage("Memory cleared.", 'system');
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Validates status on load
    updateFileList();
});
