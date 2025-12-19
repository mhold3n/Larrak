// Basic DAG Renderer (Dependency-Free)

class GraphRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth || 800;
        this.height = this.container.clientHeight || 600;
        
        this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.svg.setAttribute("width", "100%");
        this.svg.setAttribute("height", "100%");
        this.svg.style.cursor = "grab";
        this.container.style.overflow = "hidden";
        this.container.appendChild(this.svg);

        // Pan/zoom state
        this.zoom = 1;
        this.translate = { x: 0, y: 0 };
        this.isPanning = false;
        this.panStart = { x: 0, y: 0 };
        this.translateStart = { x: 0, y: 0 };

        // Root group to apply pan/zoom transforms
        this.rootGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        this.svg.appendChild(this.rootGroup);

        // Mouse interactions
        this.svg.addEventListener("mousedown", (e) => {
            this.isPanning = true;
            this.panStart = { x: e.clientX, y: e.clientY };
            this.translateStart = { ...this.translate };
            this.svg.style.cursor = "grabbing";
        });
        window.addEventListener("mouseup", () => {
            this.isPanning = false;
            this.svg.style.cursor = "grab";
        });
        window.addEventListener("mousemove", (e) => {
            if (!this.isPanning) return;
            const dx = e.clientX - this.panStart.x;
            const dy = e.clientY - this.panStart.y;
            this.translate = {
                x: this.translateStart.x + dx,
                y: this.translateStart.y + dy,
            };
            this.applyTransform();
        });
        this.svg.addEventListener("wheel", (e) => {
            e.preventDefault();
            const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.min(5, Math.max(0.2, this.zoom * scaleFactor));
            // Zoom to cursor position
            const rect = this.svg.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            this.translate.x = cx - (cx - this.translate.x) * (newZoom / this.zoom);
            this.translate.y = cy - (cy - this.translate.y) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.applyTransform();
        }, { passive: false });
        
        this.nodes = [];
        this.links = [];
    }

    applyTransform() {
        if (!this.rootGroup) return;
        this.rootGroup.setAttribute("transform", `translate(${this.translate.x},${this.translate.y}) scale(${this.zoom})`);
    }
    
    setData(data) {
        // Simple Layer Layout
        // 1. Identify Ranks
        // Modules are central. Artifacts are inputs or outputs.
        // Let's rely on a force-simulation concept but implemented simply or just use layers.
        
        // Let's do a simple layered approach:
        // Rank 0: Origin Artifacts
        // Rank 1: Modules using Origin
        // Rank 2: Artifacts produced by Rank 1
        // Rank 3: Modules using Rank 2
        
        const nodes = data.nodes;
        const links = data.links;
        
        // Build adjacency
        const adj = {};
        const rev_adj = {};
        nodes.forEach(n => { adj[n.id] = []; rev_adj[n.id] = []; });
        links.forEach(l => {
            if (adj[l.source]) adj[l.source].push(l.target);
            if (rev_adj[l.target]) rev_adj[l.target].push(l.source);
        });
        
        // Assign Ranks (Topological Sort simplified)
        const ranks = {};
        const visited = new Set();
        
        function getRank(nodeId) {
            if (ranks[nodeId] !== undefined) return ranks[nodeId];
            if (visited.has(nodeId)) return 0; // Cycle detected
            visited.add(nodeId);
            
            const parents = rev_adj[nodeId];
            if (!parents || parents.length === 0) {
                ranks[nodeId] = 0;
            } else {
                let maxP = 0;
                parents.forEach(p => {
                    maxP = Math.max(maxP, getRank(p));
                });
                ranks[nodeId] = maxP + 1;
            }
            return ranks[nodeId];
        }
        
        nodes.forEach(n => getRank(n.id));
        
        // Group by Rank
        const layers = [];
        nodes.forEach(n => {
            const r = ranks[n.id];
            if (!layers[r]) layers[r] = [];
            layers[r].push(n);
        });
        
        // Layout: wrap into rows of up to 5 columns (left->right), then move down
        const nodeWidth = 140;
        const nodeHeight = 50;
        const xSpacing = 220;
        const ySpacing = 140;
        const maxColumns = 5;

        nodes.forEach((n, idx) => {
            const col = idx % maxColumns;
            const row = Math.floor(idx / maxColumns);
            n.x = 50 + col * xSpacing;
            n.y = 50 + row * ySpacing;
        });

        // Compute bounds for viewBox auto-fit
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        nodes.forEach(n => {
            minX = Math.min(minX, n.x);
            minY = Math.min(minY, n.y);
            maxX = Math.max(maxX, n.x + 140);
            maxY = Math.max(maxY, n.y + 40);
        });
        const pad = 120; // extra padding to avoid clipping
        const vbX = minX - pad;
        const vbY = minY - pad;
        const vbW = (maxX - minX) + pad * 2;
        const vbH = (maxY - minY) + pad * 2;
        this.svg.setAttribute("viewBox", `${vbX} ${vbY} ${vbW} ${vbH}`);
        // Reset pan/zoom so that natural scrolling shows everything; disable auto-translate for vertical flow
        this.zoom = 1;
        this.translate = { x: 0, y: 0 };
        this.applyTransform();
        
        // Draw
        this.rootGroup.innerHTML = ""; // Clear
        
        // Ensure markers exist for arrows
        this.ensureMarkers();

        // Draw Links
        links.forEach(l => {
            const source = nodes.find(n => n.id === l.source);
            const target = nodes.find(n => n.id === l.target);
            if(source && target) {
                this.drawLink(source, target, l);
            }
        });
        
        // Draw Nodes
        nodes.forEach(n => {
            this.drawNode(n);
        });
    }
    
    ensureMarkers() {
        if (this.markersReady) return;
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");

        const mk = (id, color) => {
            const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
            marker.setAttribute("id", id);
            marker.setAttribute("markerWidth", "10");
            marker.setAttribute("markerHeight", "10");
            marker.setAttribute("refX", "10");
            marker.setAttribute("refY", "3");
            marker.setAttribute("orient", "auto");
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", "M0,0 L10,3 L0,6 Z");
            path.setAttribute("fill", color);
            marker.appendChild(path);
            defs.appendChild(marker);
        };

        mk("arrow-default", "#8b949e");
        mk("arrow-retry", "#f85149");
        this.svg.insertBefore(defs, this.rootGroup);
        this.markersReady = true;
    }

    drawNode(n) {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("transform", `translate(${n.x},${n.y})`);
        g.style.cursor = "pointer";
        // Show display_name in alert, with technical id for modules
        const displayLabel = n.display_name || n.label;
        const technicalId = n.id;
        g.onclick = () => alert(`${displayLabel}\n\nType: ${n.type}\nID: ${technicalId}`);
        
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("width", 140);
        rect.setAttribute("height", 40);
        rect.setAttribute("rx", 5);
        rect.setAttribute("fill", n.type === 'module' ? '#1f6feb' : '#238636'); // Blue for Module, Green for File
        rect.setAttribute("stroke", "#30363d");
        
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", 70);
        text.setAttribute("y", 25);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", "white");
        text.setAttribute("font-size", "12px");
        // Prefer display_name over label
        const labelText = displayLabel.length > 18 ? displayLabel.substring(0,16)+"..." : displayLabel;
        text.textContent = labelText;
        
        // Tooltip shows both display name and technical id
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = n.type === 'module' ? `${displayLabel} (${technicalId})` : displayLabel;
        g.appendChild(title);
        
        g.appendChild(rect);
        g.appendChild(text);
        this.rootGroup.appendChild(g);
    }
    
    drawLink(s, t, link) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        // Curvy line
        const x1 = s.x + 140;
        const y1 = s.y + 20;
        const x2 = t.x;
        const y2 = t.y + 20;
        
        const d = `M ${x1} ${y1} C ${(x1+x2)/2} ${y1}, ${(x1+x2)/2} ${y2}, ${x2} ${y2}`;
        path.setAttribute("d", d);
        const isRetry = link.type === "loop_retry";
        path.setAttribute("stroke", isRetry ? "#f85149" : "#8b949e");
        path.setAttribute("stroke-width", 2);
        if (isRetry) {
            path.setAttribute("stroke-dasharray", "6 4");
            path.setAttribute("marker-end", "url(#arrow-retry)");
        } else {
            path.setAttribute("marker-end", "url(#arrow-default)");
        }
        path.setAttribute("fill", "none");
        this.rootGroup.appendChild(path);

        if (link.label) {
            const tx = (x1 + x2) / 2;
            const ty = (y1 + y2) / 2 - 6;
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", tx);
            text.setAttribute("y", ty);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("fill", isRetry ? "#f85149" : "#8b949e");
            text.setAttribute("font-size", "10px");
            text.textContent = link.label;
            this.rootGroup.appendChild(text);
        }
    }
}

// Global instance
window.graphRenderer = null;

function renderGraph() {
    const container = document.getElementById('graph-container');
    if (!container) return; // Tab not active or HTML not updated
    
    container.innerHTML = ""; // Clear for new renderer
    window.graphRenderer = new GraphRenderer('graph-container');
    
    fetch('/api/graph')
        .then(r => r.json())
        .then(data => {
            window.graphRenderer.setData(data);
        });
}
