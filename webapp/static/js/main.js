// State
let pages = [];
let socket = null;

// #region agent log
function agentLog(hypothesisId, location, message, data) {
    fetch('http://127.0.0.1:7242/ingest/3c0790f3-0be0-49cd-a35b-74d95b13a271',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:location,message:message,data:data,timestamp:Date.now(),sessionId:'debug-session',runId:'pre-fix',hypothesisId:hypothesisId})}).catch(()=>{});
}
// #endregion agent log

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initWebsocket();
    loadPages();
    loadModules();
});

// --- Real-time Logic ---

function initWebsocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/log`);
    
    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'log') {
            appendLog(msg.content);
        } else if (msg.type === 'status') {
            // #region agent log
            agentLog('E','dashboard_app/static/js/main.js:socket.onmessage','Status message received',{content:msg.content,module_id:msg.module_id,error:msg.error});
            // #endregion agent log
            updateStatus(msg.content);
            if (isSequenceRunning && msg.content === 'success') {
                // Wait small delay then trigger next
                setTimeout(processQueue, 1000);
            }
        }
    };
    
    socket.onclose = () => {
        // Reconnect after 3s
        setTimeout(initWebsocket, 3000);
    };
}

function updateStatus(status) {
    const pill = document.querySelector('.status-pill');
    const label = document.getElementById('system-status');
    const dot = document.querySelector('.status-dot');
    
    label.textContent = status.toUpperCase();
    
    if (status === 'running') {
        dot.classList.add('active');
        dot.style.background = 'var(--accent-color)';
    } else if (status === 'success') {
        dot.classList.remove('active');
        dot.style.background = 'var(--success)';
    } else if (status === 'failure') {
        dot.classList.remove('active');
        dot.style.background = 'var(--danger)';
    }
}

function appendLog(text) {
    const consoleDiv = document.getElementById('console-output');
    if (!consoleDiv) return;
    
    const line = document.createElement('div');
    line.textContent = text;
    line.style.fontFamily = 'monospace';
    line.style.fontSize = '0.85em';
    line.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
    consoleDiv.appendChild(line);
    
    // Auto scroll
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
}

// --- Navigation ---

function initTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tab = item.dataset.tab;
            
            // Update Sidebar
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Update Title
            document.getElementById('page-title').textContent = item.querySelector('span').textContent;
            
            // Update View
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            
            const target = document.getElementById(`tab-${tab}`);
            if(target) {
                target.classList.remove('hidden');
                if (tab === 'graph') {
                    renderGraph();
                }
                if (tab === 'docs') {
                    loadDocsTree();
                }
                if (tab === 'history') {
                    loadRunHistory();
                }
            }
        });
    });
    
    document.getElementById('refresh-btn').addEventListener('click', loadPages);
}

function openViewer(url, title) {
    const frame = document.getElementById('content-frame');
    frame.src = url;
    
    // Switch to viewer tab (virtual tab)
    document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
    document.getElementById('tab-viewer').classList.remove('hidden');
    document.getElementById('page-title').textContent = title;
}

// --- Data Loading ---

async function loadPages() {
    try {
        const res = await fetch('/api/pages');
        pages = await res.json();
        render();
    } catch (e) {
        console.error("Failed to load pages", e);
    }
}

async function loadModules() {
    try {
        const res = await fetch('/api/modules');
        const modules = await res.json();
        renderRunControl(modules);
    } catch (e) {
        console.error("Failed to load modules", e);
    }
}

// --- Run Control ---

// Queue State
let runQueue = [];
let isSequenceRunning = false;

function renderRunControl(modules) {
    const container = document.getElementById('tab-runs');
    container.innerHTML = `
        <div style="display:flex; gap:20px; height:100%;">
            <div style="flex:1; overflow-y:auto;" id="modules-list">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2>Run Control</h2>
                    <button onclick="runSequence()" 
                            id="run-all-btn"
                            style="padding:8px 16px; background:var(--accent-color); border:none; color:white; border-radius:4px; cursor:pointer; font-weight:600;">
                        ▶ Run All Sequence
                    </button>
                </div>
                <!-- Progress Indicator -->
                <div id="sequence-progress" style="display:none; margin-bottom:10px; background:#161b22; padding:10px; border-radius:4px;">
                    <span id="sequence-status" style="color:#8b949e; font-size:0.9em;">Running sequence...</span>
                </div>
                <div class="grid" id="modules-grid"></div>
            </div>
            <div style="flex:1; display:flex; flex-direction:column; background:#0d1117; border-left:1px solid #30363d; padding:20px;">
                <h3>Live Logs</h3>
                <div id="console-output" style="flex:1; overflow-y:auto; background:black; padding:10px; border-radius:6px; font-family:monospace;">
                    <div style="color:#586069">Waiting for logs...</div>
                </div>
            </div>
        </div>
    `;
    
    const grid = document.getElementById('modules-grid');
    
    modules.forEach(mod => {
        const card = document.createElement('div');
        card.className = 'card';
        // Add ID for finding later
        card.dataset.moduleId = mod.module_id;
        
        // ... (rest of card params rendering is same) ...
        let paramsHtml = '';
        if (mod.params) {
            mod.params.forEach(p => {
                if(p.type === 'bool') {
                   paramsHtml += `
                    <div style="margin-bottom:10px;">
                        <label style="display:flex; align-items:center; gap:10px; font-size:0.9em;">
                            <input type="checkbox" name="${p.name}" ${p.default ? 'checked' : ''}>
                            ${p.name}
                        </label>
                        <div style="font-size:0.8em; color:#586069; margin-left:24px;">${p.description}</div>
                    </div>
                   `;
                } else {
                   paramsHtml += `
                    <div style="margin-bottom:10px;">
                        <label style="display:block; font-size:0.9em; margin-bottom:4px;">${p.name}</label>
                        <input type="${p.type === 'int' || p.type === 'float' ? 'number' : 'text'}" 
                               name="${p.name}" 
                               value="${p.default}" 
                               step="${p.type === 'float' ? '0.1' : '1'}"
                               style="width:100%; box-sizing:border-box; padding:6px; background:#0d1117; border:1px solid #30363d; color:white; border-radius:4px;">
                        <div style="font-size:0.8em; color:#586069; margin-top:4px;">${p.description}</div>
                    </div>
                   `;
                }
            });
        }
        
        card.innerHTML = `
            <div class="card-header">
                <span class="tag dashboard">${mod.owner}</span>
            </div>
            <h3>${mod.module_id}</h3>
            <p>${mod.description}</p>
            <form id="form-${mod.module_id}" onsubmit="return false;">
                ${paramsHtml}
                <button onclick="runModule('${mod.module_id}')" 
                        class="run-btn"
                        style="width:100%; padding:8px; background:var(--success); border:none; color:white; border-radius:4px; cursor:pointer; font-weight:600; margin-top:10px;">
                    Run Module
                </button>
            </form>
        `;
        grid.appendChild(card);
    });
}

function runSequence() {
    // 1. Collect all module IDs in order from the grid
    const cards = document.querySelectorAll('.card[data-module-id]');
    runQueue = Array.from(cards).map(c => c.dataset.moduleId);
    
    if (runQueue.length === 0) {
        alert("No modules to run!");
        return;
    }
    
    // 2. Start UI Mode
    isSequenceRunning = true;
    document.getElementById('run-all-btn').disabled = true;
    document.getElementById('run-all-btn').style.opacity = 0.5;
    document.getElementById('sequence-progress').style.display = 'block';
    
    appendLog(`--- STARTING SEQUENCE: ${runQueue.join(' -> ')} ---`);
    processQueue();
}

function processQueue() {
    if (runQueue.length === 0) {
        appendLog("--- SEQUENCE COMPLETE ---");
        isSequenceRunning = false;
        document.getElementById('run-all-btn').disabled = false;
        document.getElementById('run-all-btn').style.opacity = 1;
        document.getElementById('sequence-status').textContent = "Sequence completed.";
        return;
    }
    
    const nextModuleId = runQueue.shift();
    document.getElementById('sequence-status').textContent = `Running ${nextModuleId} (${runQueue.length} remaining)...`;
    runModule(nextModuleId, true); // true = silent alert
}

function runModule(moduleId, silent = false) {
    const form = document.getElementById(`form-${moduleId}`);
    const inputs = form.querySelectorAll('input');
    const params = {};
    
    inputs.forEach(inp => {
        if(inp.type === 'checkbox') {
            params[inp.name] = inp.checked;
        } else {
            params[inp.name] = inp.value;
        }
    });

    // #region agent log
    agentLog('D','dashboard_app/static/js/main.js:runModule','Starting runModule fetch',{module_id:moduleId,param_keys:Object.keys(params)});
    // #endregion agent log
    
    fetch('/api/runs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ module_id: moduleId, params: params })
    })
    .then(r => r.json())
    .then(data => {
        if(!silent) alert(`Started run for ${moduleId}`);
        document.getElementById('system-status').textContent = "Running...";
        document.querySelector('.status-dot').classList.add('active');
        
        // Highlight active card
        document.querySelectorAll('.card').forEach(c => c.style.borderColor = 'rgba(255,255,255,0.1)');
        const activeCard = document.querySelector(`.card[data-module-id="${moduleId}"]`);
        if(activeCard) activeCard.style.borderColor = 'var(--accent-color)';
    })
    .catch(e => {
        alert("Error starting run");
        // If in sequence, maybe authorize skip? For now, we halt or continue?
        // Let's halt if API call fails.
        isSequenceRunning = false;
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadPages();
    loadModules();
});

// --- Rendering ---

function render() {
    const recentGrid = document.getElementById('recent-grid');
    const dashboardGrid = document.getElementById('dashboard-grid');
    const reportGrid = document.getElementById('report-grid');
    
    recentGrid.innerHTML = '';
    dashboardGrid.innerHTML = '';
    reportGrid.innerHTML = '';
    
    // Sort by time
    pages.sort((a,b) => b.timestamp - a.timestamp);
    
    // Populate Overview (Top 4)
    pages.slice(0, 4).forEach(page => {
        recentGrid.appendChild(createCard(page));
    });
    
    // Populate Tabs
    pages.forEach(page => {
        const card = createCard(page);
        if (page.type === 'module_dashboard') {
            dashboardGrid.appendChild(card);
        } else if (page.type === 'report') {
            reportGrid.appendChild(card);
        }
    });
}

function createCard(page) {
    const card = document.createElement('div');
    card.className = 'card';
    card.onclick = () => openViewer(page.path, page.title);
    
    const date = new Date(page.timestamp * 1000).toLocaleString();
    
    let tagClass = page.type === 'module_dashboard' ? 'dashboard' : 'report';
    let tagLabel = page.type === 'module_dashboard' ? 'Module' : 'Artifact';
    
    card.innerHTML = `
        <div class="card-header">
            <span class="tag ${tagClass}">${tagLabel}</span>
        </div>
        <h3>${page.title}</h3>
        <p>${date}</p>
        <div style="font-size:0.8em; color:#586069; word-break:break-all;">
            ${page.page_id}
        </div>
    `;
    return card;
}

// --- Documentation ---

let docsTreeLoaded = false;

async function loadDocsTree() {
    // Only load once
    if (docsTreeLoaded) return;
    
    try {
        const res = await fetch('/api/docs/tree');
        const tree = await res.json();
        renderDocsTree(tree);
        docsTreeLoaded = true;
    } catch (e) {
        console.error("Failed to load docs tree", e);
        document.getElementById('docs-tree-content').innerHTML = 
            '<div style="padding:10px; color:var(--danger);">Failed to load documentation tree.</div>';
    }
}

function renderDocsTree(nodes, container = null, level = 0) {
    if (!container) {
        container = document.getElementById('docs-tree-content');
        container.innerHTML = '';
    }
    
    nodes.forEach(node => {
        const item = document.createElement('div');
        item.className = 'docs-tree-item';
        item.style.paddingLeft = `${12 + level * 16}px`;
        
        if (node.type === 'folder') {
            // Folder node with expand/collapse
            item.innerHTML = `
                <span class="docs-tree-chevron">▶</span>
                <span class="docs-tree-icon">📁</span>
                <span class="docs-tree-label">${node.name}</span>
            `;
            item.classList.add('docs-tree-folder');
            
            const childContainer = document.createElement('div');
            childContainer.className = 'docs-tree-children';
            childContainer.style.display = 'none';
            
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                const chevron = item.querySelector('.docs-tree-chevron');
                const isExpanded = childContainer.style.display !== 'none';
                
                if (isExpanded) {
                    childContainer.style.display = 'none';
                    chevron.textContent = '▶';
                    chevron.classList.remove('expanded');
                } else {
                    childContainer.style.display = 'block';
                    chevron.textContent = '▼';
                    chevron.classList.add('expanded');
                }
            });
            
            container.appendChild(item);
            container.appendChild(childContainer);
            
            // Recursively render children
            if (node.children && node.children.length > 0) {
                renderDocsTree(node.children, childContainer, level + 1);
            }
        } else {
            // File node
            item.innerHTML = `
                <span class="docs-tree-chevron" style="visibility:hidden">▶</span>
                <span class="docs-tree-icon">📄</span>
                <span class="docs-tree-label">${node.name}</span>
            `;
            item.classList.add('docs-tree-file');
            item.dataset.path = node.path;
            
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                // Remove active class from all items
                document.querySelectorAll('.docs-tree-file').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
                renderDocContent(node.path);
            });
            
            container.appendChild(item);
        }
    });
}

async function renderDocContent(path) {
    const contentDiv = document.getElementById('docs-content');
    contentDiv.innerHTML = '<div class="docs-loading">Loading...</div>';
    
    try {
        const res = await fetch(`/api/docs/render?path=${encodeURIComponent(path)}`);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        
        contentDiv.innerHTML = `
            <div class="docs-article">
                <h1 class="docs-title">${data.title}</h1>
                <div class="docs-body">${data.html}</div>
            </div>
        `;
    } catch (e) {
        console.error("Failed to render doc", e);
        contentDiv.innerHTML = `
            <div class="docs-error">
                <h2>Error</h2>
                <p>Failed to load document: ${path}</p>
            </div>
        `;
    }
}

// --- Run History ---

let runHistoryLoaded = false;

async function loadRunHistory() {
    const timeline = document.getElementById('history-timeline');
    const statsDiv = document.getElementById('history-stats');
    
    timeline.innerHTML = '<div style="color:var(--text-secondary); padding:20px;">Loading run history...</div>';
    
    try {
        // Load both runs and stats
        const [runsRes, statsRes] = await Promise.all([
            fetch('/api/runs/history?limit=30'),
            fetch('/api/runs/stats')
        ]);
        
        const runs = await runsRes.json();
        const stats = await statsRes.json();
        
        renderRunStats(stats, statsDiv);
        renderRunHistory(runs, timeline);
        runHistoryLoaded = true;
    } catch (e) {
        console.error("Failed to load run history", e);
        timeline.innerHTML = '<div style="color:var(--danger); padding:20px;">Failed to load run history.</div>';
    }
}

function renderRunStats(stats, container) {
    container.innerHTML = `
        <div class="stats-row">
            <div class="stat-item">
                <span class="stat-value">${stats.total}</span>
                <span class="stat-label">Total Runs</span>
            </div>
            <div class="stat-item stat-success">
                <span class="stat-value">${stats.success}</span>
                <span class="stat-label">Successful</span>
            </div>
            <div class="stat-item stat-failure">
                <span class="stat-value">${stats.failure}</span>
                <span class="stat-label">Failed</span>
            </div>
            <div class="stat-item stat-running">
                <span class="stat-value">${stats.running}</span>
                <span class="stat-label">Running</span>
            </div>
        </div>
    `;
}

function renderRunHistory(runs, container) {
    if (runs.length === 0) {
        container.innerHTML = '<div style="color:var(--text-secondary); padding:20px; text-align:center;">No runs found. Trigger a run from the Run Control tab.</div>';
        return;
    }
    
    container.innerHTML = '';
    
    runs.forEach(run => {
        const card = createRunCard(run);
        container.appendChild(card);
    });
}

function createRunCard(run) {
    const card = document.createElement('div');
    card.className = 'run-card';
    card.dataset.runId = run.run_id;
    
    // Status styling
    let statusClass = 'status-unknown';
    let statusIcon = '○';
    if (run.status === 'SUCCESS') {
        statusClass = 'status-success';
        statusIcon = '✓';
    } else if (run.status === 'FAILURE') {
        statusClass = 'status-failure';
        statusIcon = '✗';
    } else if (run.status === 'RUNNING') {
        statusClass = 'status-running';
        statusIcon = '⟳';
    }
    
    // Format time
    let startTimeStr = 'Unknown';
    if (run.start_time) {
        try {
            const dt = new Date(run.start_time);
            startTimeStr = dt.toLocaleString();
        } catch(e) {
            startTimeStr = run.start_time;
        }
    }
    
    card.innerHTML = `
        <div class="run-card-header">
            <span class="run-status-badge ${statusClass}">${statusIcon} ${run.status}</span>
            <span class="run-module">${run.module_id}</span>
        </div>
        <div class="run-card-body">
            <div class="run-meta">
                <span class="run-time">Started: ${startTimeStr}</span>
                <span class="run-duration">Duration: ${run.duration}</span>
            </div>
        </div>
        <div class="run-card-footer" id="run-details-${run.run_id}" style="display:none;">
            <div class="run-artifacts-loading">Loading artifacts...</div>
        </div>
    `;
    
    card.addEventListener('click', () => expandRunDetails(run.run_id));
    
    return card;
}

async function expandRunDetails(runId) {
    const footer = document.getElementById(`run-details-${runId}`);
    if (!footer) return;
    
    // Toggle visibility
    if (footer.style.display === 'block') {
        footer.style.display = 'none';
        return;
    }
    
    footer.style.display = 'block';
    
    try {
        const res = await fetch(`/api/runs/${runId}`);
        const details = await res.json();
        
        let artifactsHtml = '<div class="no-artifacts">No artifacts linked to this run.</div>';
        
        if (details.artifacts && details.artifacts.length > 0) {
            artifactsHtml = '<div class="run-artifacts-list">';
            details.artifacts.forEach(art => {
                // Create clickable link for the artifact
                const path = art.path || '';
                // Handle both forward and backslashes in paths
                const filename = path.replace(/\\/g, '/').split('/').pop() || art.artifact_id;
                // Normalize path for web
                const normalizedPath = path.replace(/\\/g, '/');
                const webPath = normalizedPath.startsWith('/') ? `/artifacts${normalizedPath}` : `/artifacts/${normalizedPath}`;
                
                artifactsHtml += `
                    <div class="artifact-item" onclick="openViewer('${webPath}', '${filename}'); event.stopPropagation();">
                        <span class="artifact-icon">📄</span>
                        <span class="artifact-name">${filename}</span>
                        <span class="artifact-role">${art.role || ''}</span>
                    </div>
                `;
            });
            artifactsHtml += '</div>';
        }
        
        // Show env params if any
        let paramsHtml = '';
        if (details.env && Object.keys(details.env).length > 0) {
            const params = Object.entries(details.env)
                .filter(([k, v]) => k.startsWith('LARRAK_') && k !== 'LARRAK_RUN_ID')
                .map(([k, v]) => `<span class="param-tag">${k.replace('LARRAK_', '')}: ${v}</span>`)
                .join(' ');
            if (params) {
                paramsHtml = `<div class="run-params">${params}</div>`;
            }
        }
        
        footer.innerHTML = `
            ${paramsHtml}
            <div class="run-artifacts-header">Artifacts (${details.artifacts?.length || 0})</div>
            ${artifactsHtml}
        `;
    } catch (e) {
        console.error("Failed to load run details", e);
        footer.innerHTML = '<div class="run-artifacts-error">Failed to load details.</div>';
    }
}

