document.addEventListener('DOMContentLoaded', () => {
    // ─── DOM Elements ───
    const form = document.getElementById('analyze-form');
    const inputSource = document.getElementById('video-source');
    const selectLanguage = document.getElementById('language-select');
    const btnAnalyse = document.getElementById('btn-analyse');
    
    const viewEmpty = document.getElementById('view-empty');
    const viewRunning = document.getElementById('view-running');
    const viewDashboard = document.getElementById('view-dashboard');
    
    const runningStepLabel = document.getElementById('running-step-label');
    const runningSubLabel = document.getElementById('running-sub-label');
    
    const dashboardTitleText = document.getElementById('dashboard-title-text');
    const overviewSummary = document.getElementById('overview-summary-text');
    const overviewActions = document.getElementById('overview-actions-text');
    const insightsDecisions = document.getElementById('insights-decisions-text');
    const insightsQuestions = document.getElementById('insights-questions-text');
    const transcriptBody = document.getElementById('transcript-body-text');
    
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessagesContainer = document.getElementById('chat-messages-container');
    const chatWelcomeMsg = document.getElementById('chat-welcome-msg');
    
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // ─── State Variables ───
    let pollingInterval = null;
    const stepOrder = ['audio', 'transcript', 'title', 'summary', 'extract', 'rag'];
    const stepNames = {
        audio: 'Audio Processing',
        transcript: 'Audio Transcription',
        title: 'Title Generation',
        summary: 'Summarisation',
        extract: 'Insights Extraction',
        rag: 'RAG Database Indexing'
    };
    const stepDescriptions = {
        audio: 'Downloading YouTube audio or converting local video file format...',
        transcript: 'Transcribing speech to text locally using Whisper AI models...',
        title: 'Synthesizing short descriptive and professional titles...',
        summary: 'Constructing map-reduce text summary layers of transcript...',
        extract: 'Extracting action items, key decisions, and discussion questions...',
        rag: 'Generating sentence-transformer vectors and indexing to Chroma DB...'
    };

    // ─── Tab Switching Logic ───
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // ─── Transition UI Views ───
    function showView(viewName) {
        viewEmpty.classList.remove('active');
        viewRunning.classList.remove('active');
        viewDashboard.classList.remove('active');
        
        if (viewName === 'empty') viewEmpty.classList.add('active');
        if (viewName === 'running') viewRunning.classList.add('active');
        if (viewName === 'dashboard') viewDashboard.classList.add('active');
    }

    // ─── Poll Pipeline Status ───
    function startPolling() {
        if (pollingInterval) clearInterval(pollingInterval);
        
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.status === 'running') {
                    updateTimeline(data.steps);
                } else if (data.status === 'done') {
                    clearInterval(pollingInterval);
                    updateTimeline(data.steps);
                    await loadResults();
                } else if (data.status === 'error') {
                    clearInterval(pollingInterval);
                    alert(`Analysis Pipeline Error: ${data.error}`);
                    resetTimeline();
                    showView('empty');
                    btnAnalyse.disabled = false;
                }
            } catch (err) {
                console.error('Error fetching status:', err);
            }
        }, 1000);
    }

    // ─── Update Timeline UI (Active vs Next) ───
    function updateTimeline(steps) {
        let activeIndex = -1;
        
        // Mark Completed and Active steps
        stepOrder.forEach((stepKey, idx) => {
            const el = document.getElementById(`step-${stepKey}`);
            if (!el) return;
            
            const state = steps[stepKey] || 'pending';
            el.classList.remove('done', 'active', 'next');
            
            if (state === 'done') {
                el.classList.add('done');
            } else if (state === 'active') {
                el.classList.add('active');
                activeIndex = idx;
                
                // Update loader view labels
                runningStepLabel.textContent = stepNames[stepKey];
                runningSubLabel.textContent = stepDescriptions[stepKey];
            }
        });
        
        // Determine and Style the "Next" Step
        let nextIndex = -1;
        if (activeIndex !== -1) {
            nextIndex = activeIndex + 1;
        } else {
            // Find first pending step
            nextIndex = stepOrder.findIndex(stepKey => (steps[stepKey] || 'pending') === 'pending');
        }
        
        if (nextIndex !== -1 && nextIndex < stepOrder.length) {
            const nextStepKey = stepOrder[nextIndex];
            const nextEl = document.getElementById(`step-${nextStepKey}`);
            if (nextEl && !nextEl.classList.contains('active') && !nextEl.classList.contains('done')) {
                nextEl.classList.add('next');
            }
        }
    }

    function resetTimeline() {
        stepOrder.forEach(stepKey => {
            const el = document.getElementById(`step-${stepKey}`);
            if (el) el.classList.remove('done', 'active', 'next');
        });
    }

    // ─── Trigger Pipeline Run ───
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const source = inputSource.value.trim();
        const language = selectLanguage.value;
        
        if (!source) return;
        
        btnAnalyse.disabled = true;
        resetTimeline();
        showView('running');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source, language })
            });
            const result = await response.json();
            
            if (response.ok) {
                startPolling();
            } else {
                alert(`Error starting pipeline: ${result.detail || 'Unknown error'}`);
                showView('empty');
                btnAnalyse.disabled = false;
            }
        } catch (err) {
            console.error('Submit analysis error:', err);
            alert('Failed to connect to the server.');
            showView('empty');
            btnAnalyse.disabled = false;
        }
    });

    // ─── Load Results from Server ───
    async function loadResults() {
        try {
            const response = await fetch('/api/result');
            const data = await response.json();
            
            if (response.ok && data.result) {
                const res = data.result;
                dashboardTitleText.textContent = res.title || 'Analysis Results';
                overviewSummary.innerHTML = formatMarkdown(res.summary);
                overviewActions.innerHTML = formatMarkdown(res.action_items);
                insightsDecisions.innerHTML = formatMarkdown(res.decisions);
                insightsQuestions.innerHTML = formatMarkdown(res.questions);
                transcriptBody.textContent = res.transcript;
                
                // Clear old chat logs
                const messages = chatMessagesContainer.querySelectorAll('.chat-msg');
                messages.forEach(m => m.remove());
                chatWelcomeMsg.style.display = 'flex';
                
                showView('dashboard');
            } else {
                alert('Failed to retrieve pipeline results.');
                showView('empty');
            }
        } catch (err) {
            console.error('Error loading results:', err);
            alert('Error loading results.');
            showView('empty');
        } finally {
            btnAnalyse.disabled = false;
        }
    }

    // ─── Q&A Chat Client ───
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;
        
        // Hide welcome block
        chatWelcomeMsg.style.display = 'none';
        
        // Append User Bubble
        appendChatBubble('user', question);
        chatInput.value = '';
        
        // Append Temporary Typing Indicator
        const typingId = appendChatBubble('bot', '🤖 Thinking...');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            const data = await response.json();
            
            // Remove typing bubble and append official answer
            const typingBubble = document.getElementById(typingId);
            if (typingBubble) typingBubble.remove();
            
            if (response.ok) {
                appendChatBubble('bot', data.answer);
            } else {
                appendChatBubble('bot', '⚠️ Failed to connect to server backend.');
            }
        } catch (err) {
            console.error('Chat Q&A error:', err);
            const typingBubble = document.getElementById(typingId);
            if (typingBubble) typingBubble.remove();
            appendChatBubble('bot', '⚠️ Error contacting RAG database.');
        }
    });

    // Append message bubble to chat container
    function appendChatBubble(sender, text) {
        const msgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender === 'user' ? 'user-msg' : 'bot-msg'}`;
        msgDiv.id = msgId;
        msgDiv.style.alignItems = sender === 'user' ? 'flex-end' : 'flex-start';
        
        const labelSpan = document.createElement('span');
        labelSpan.className = `chat-label ${sender === 'user' ? 'user-label' : 'bot-label'}`;
        labelSpan.textContent = sender === 'user' ? 'You' : '🤖 Assistant';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = `chat-bubble ${sender === 'user' ? 'user-bubble' : 'bot-bubble'}`;
        bubbleDiv.innerHTML = text.replace(/\n/g, '<br>');
        
        msgDiv.appendChild(labelSpan);
        msgDiv.appendChild(bubbleDiv);
        chatMessagesContainer.appendChild(msgDiv);
        
        // Auto-scroll
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        
        return msgId;
    }

    // Helper: Basic Markdown to HTML converter (bold and list rendering)
    function formatMarkdown(text) {
        if (!text) return '';
        let html = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
        return html;
    }
});
