/**
 * Workflow Visualizer Module
 * 
 * Implements trustless agent loop visualization using D3.js force-directed graph.
 * Displays 10-step pipeline with real-time state updates, interactive nodes,
 * and branching decision points (MINT/REJECT/REVIEW, on-chain/off-chain routing).
 * 
 * @module WorkflowVisualizer
 * @requires d3.js v7
 * @version 1.0.0
 */

class WorkflowVisualizer {
    /**
     * Initialize workflow visualizer
     * @param {string} containerId - DOM element ID for SVG container
     * @param {Object} options - Configuration options
     */
    constructor(containerId = 'workflowGraph', options = {}) {
        console.log(`[WorkflowVisualizer] Entry: constructor(containerId=${containerId})`);
        
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            width: options.width || 800,
            height: options.height || 600,
            nodeRadius: options.nodeRadius || 30,
            layoutType: options.layoutType || 'vertical', // 'vertical' or 'horizontal'
            animationDuration: options.animationDuration || 500,
            ...options
        };
        
        // Workflow state
        this.currentStep = null;
        this.correlationId = null;
        this.workflowData = null;
        
        // D3 elements
        this.svg = null;
        this.simulation = null;
        
        // Define 10-step workflow structure
        this.workflowSteps = [
            { id: 'upload', label: 'Upload\nBVH/FBX', icon: '📤', group: 'input' },
            { id: 'ingest', label: 'Motion\nIngest', icon: '⚙️', group: 'processing' },
            { id: 'gemini', label: 'Gemini\nAnalysis', icon: '🤖', group: 'ai' },
            { id: 'oracle', label: 'Oracle\nDecision', icon: '🔮', group: 'ai', branching: true },
            { id: 'pack', label: 'Canonical\nPack', icon: '📦', group: 'processing', conditional: true },
            { id: 'routing', label: 'Tx\nRouting', icon: '🔀', group: 'routing', branching: true, conditional: true },
            { id: 'arc_tx', label: 'Arc\nTransaction', icon: '⛓️', group: 'blockchain', conditional: true },
            { id: 'circle_tx', label: 'Circle\nPayment', icon: '💰', group: 'payment', conditional: true },
            { id: 'commerce', label: 'Usage\nMeter', icon: '📊', group: 'processing', conditional: true },
            { id: 'payouts', label: 'Payout\nDistribution', icon: '💸', group: 'final', conditional: true }
        ];
        
        // Define edges (workflow connections)
        this.workflowEdges = [
            { source: 'upload', target: 'ingest', label: 'file_uri' },
            { source: 'ingest', target: 'gemini', label: 'tensor_hash' },
            { source: 'gemini', target: 'oracle', label: 'descriptors' },
            // Oracle branching
            { source: 'oracle', target: 'pack', label: 'MINT', condition: 'mint' },
            { source: 'oracle', target: 'pack', label: 'REJECT', condition: 'reject', style: 'dashed' },
            { source: 'oracle', target: 'pack', label: 'REVIEW', condition: 'review', style: 'dashed' },
            // Routing branching
            { source: 'pack', target: 'routing', label: 'pack_hash' },
            { source: 'routing', target: 'arc_tx', label: 'on-chain', condition: 'on-chain' },
            { source: 'routing', target: 'circle_tx', label: 'off-chain', condition: 'off-chain' },
            { source: 'routing', target: 'arc_tx', label: 'hybrid', condition: 'hybrid' },
            { source: 'routing', target: 'circle_tx', label: 'hybrid', condition: 'hybrid' },
            // Convergence
            { source: 'arc_tx', target: 'commerce', label: 'tx_hash' },
            { source: 'circle_tx', target: 'commerce', label: 'payment_id' },
            { source: 'commerce', target: 'payouts', label: 'usage_data' }
        ];
        
        console.log(`[WorkflowVisualizer] Exit: constructor - Initialized with ${this.workflowSteps.length} steps`);
    }
    
    /**
     * Initialize and render the workflow graph
     * @returns {Promise<void>}
     */
    async init() {
        console.log('[WorkflowVisualizer] Entry: init()');
        
        if (!this.container) {
            console.error(`[WorkflowVisualizer] Container #${this.containerId} not found`);
            return;
        }
        
        try {
            // Create SVG canvas
            this.createSVG();
            
            // Setup force simulation
            this.setupSimulation();
            
            // Render workflow
            this.render();
            
            // Setup event listeners
            this.setupEventListeners();
            
            console.log('[WorkflowVisualizer] Exit: init() - Workflow graph initialized successfully');
        } catch (error) {
            console.error('[WorkflowVisualizer] Error in init():', error);
            throw error;
        }
    }
    
    /**
     * Create SVG element and setup groups
     * @private
     */
    createSVG() {
        console.log('[WorkflowVisualizer] Entry: createSVG()');
        
        // Clear existing content
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.config.width} ${this.config.height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                this.svg.select('.graph-group').attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group for graph elements
        this.graphGroup = this.svg.append('g')
            .attr('class', 'graph-group');
        
        // Create groups for edges and nodes (order matters for z-index)
        this.edgeGroup = this.graphGroup.append('g').attr('class', 'edges');
        this.nodeGroup = this.graphGroup.append('g').attr('class', 'nodes');
        
        console.log('[WorkflowVisualizer] Exit: createSVG()');
    }
    
    /**
     * Setup D3 force simulation
     * @private
     */
    setupSimulation() {
        console.log('[WorkflowVisualizer] Entry: setupSimulation()');
        
        // Create nodes array from workflow steps
        this.nodes = this.workflowSteps.map((step, i) => ({
            ...step,
            x: this.calculateInitialX(i),
            y: this.calculateInitialY(i),
            state: 'pending' // pending, processing, success, error, review
        }));
        
        // Create links array from workflow edges
        this.links = this.workflowEdges.map(edge => ({
            source: this.nodes.find(n => n.id === edge.source),
            target: this.nodes.find(n => n.id === edge.target),
            ...edge
        }));
        
        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-500))
            .force('center', d3.forceCenter(this.config.width / 2, this.config.height / 2))
            .force('collision', d3.forceCollide().radius(this.config.nodeRadius + 10))
            .on('tick', () => this.updatePositions());
        
        console.log(`[WorkflowVisualizer] Exit: setupSimulation() - ${this.nodes.length} nodes, ${this.links.length} links`);
    }
    
    /**
     * Calculate initial X position for node (vertical layout)
     * @private
     */
    calculateInitialX(index) {
        const cols = 3; // 3 columns for vertical layout
        const col = index % cols;
        const spacing = this.config.width / (cols + 1);
        return spacing * (col + 1);
    }
    
    /**
     * Calculate initial Y position for node (vertical layout)
     * @private
     */
    calculateInitialY(index) {
        const rows = Math.ceil(this.workflowSteps.length / 3);
        const row = Math.floor(index / 3);
        const spacing = this.config.height / (rows + 1);
        return spacing * (row + 1);
    }
    
    /**
     * Render workflow graph (nodes and edges)
     * @private
     */
    render() {
        console.log('[WorkflowVisualizer] Entry: render()');
        
        // Render edges
        this.renderEdges();
        
        // Render nodes
        this.renderNodes();
        
        console.log('[WorkflowVisualizer] Exit: render()');
    }
    
    /**
     * Render edges (links between nodes)
     * @private
     */
    renderEdges() {
        const edges = this.edgeGroup.selectAll('.edge')
            .data(this.links)
            .join('g')
            .attr('class', 'edge');
        
        // Draw edge lines
        edges.append('path')
            .attr('class', d => `edge-path ${d.style || 'solid'}`)
            .attr('fill', 'none')
            .attr('stroke', '#999')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead)');
        
        // Add edge labels
        edges.append('text')
            .attr('class', 'edge-label')
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#666')
            .text(d => d.label);
        
        // Define arrowhead marker
        this.svg.append('defs')
            .append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 35)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');
    }
    
    /**
     * Render nodes (workflow steps)
     * @private
     */
    renderNodes() {
        const nodes = this.nodeGroup.selectAll('.node')
            .data(this.nodes)
            .join('g')
            .attr('class', 'node')
            .attr('cursor', 'pointer')
            .call(this.dragBehavior());
        
        // Draw node circles
        nodes.append('circle')
            .attr('r', this.config.nodeRadius)
            .attr('class', d => `node-circle ${d.group}`)
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 3);
        
        // Add node icons
        nodes.append('text')
            .attr('class', 'node-icon')
            .attr('text-anchor', 'middle')
            .attr('dy', '-0.2em')
            .attr('font-size', '24px')
            .text(d => d.icon);
        
        // Add node labels
        nodes.append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .attr('dy', this.config.nodeRadius + 15)
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .selectAll('tspan')
            .data(d => d.label.split('\n'))
            .join('tspan')
            .attr('x', 0)
            .attr('dy', (d, i) => i === 0 ? 0 : '1.1em')
            .text(d => d);
        
        // Add click event handlers
        nodes.on('click', (event, d) => this.handleNodeClick(event, d));
    }
    
    /**
     * Get node fill color based on state
     * @private
     */
    getNodeColor(node) {
        const stateColors = {
            'pending': '#95a5a6',
            'processing': '#f39c12',
            'success': '#27ae60',
            'error': '#e74c3c',
            'review': '#e67e22'
        };
        return stateColors[node.state] || '#95a5a6';
    }
    
    /**
     * Setup drag behavior for nodes
     * @private
     */
    dragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    /**
     * Update node and edge positions on simulation tick
     * @private
     */
    updatePositions() {
        // Update node positions
        this.nodeGroup.selectAll('.node')
            .attr('transform', d => `translate(${d.x},${d.y})`);
        
        // Update edge positions
        this.edgeGroup.selectAll('.edge-path')
            .attr('d', d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
            });
        
        // Update edge label positions
        this.edgeGroup.selectAll('.edge-label')
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2);
    }
    
    /**
     * Handle node click event - show detail modal
     * @private
     */
    handleNodeClick(event, node) {
        console.log(`[WorkflowVisualizer] Node clicked: ${node.id}`, node);
        
        // Show appropriate modal based on node type
        if (node.id === 'oracle') {
            this.showOracleModal(node);
        } else if (node.id === 'routing') {
            this.showRoutingModal(node);
        } else {
            this.showStepDetails(node);
        }
    }
    
    /**
     * Show Oracle decision modal
     * @private
     */
    showOracleModal(node) {
        console.log('[WorkflowVisualizer] Showing Oracle modal');
        const modal = document.getElementById('oracleModal');
        if (modal) {
            modal.style.display = 'flex';
            
            // Populate modal with data (if available)
            if (this.workflowData && this.workflowData.oracle) {
                this.populateOracleModal(this.workflowData.oracle);
            }
        }
    }
    
    /**
     * Show transaction routing modal
     * @private
     */
    showRoutingModal(node) {
        console.log('[WorkflowVisualizer] Showing Routing modal');
        const modal = document.getElementById('routingModal');
        if (modal) {
            modal.style.display = 'flex';
            
            // Populate modal with data (if available)
            if (this.workflowData && this.workflowData.routing) {
                this.populateRoutingModal(this.workflowData.routing);
            }
        }
    }
    
    /**
     * Show generic step details
     * @private
     */
    showStepDetails(node) {
        console.log(`[WorkflowVisualizer] Showing details for step: ${node.id}`);
        // TODO: Implement generic step detail view
        alert(`Step: ${node.label}\nState: ${node.state}\nData: ${JSON.stringify(node.data || {}, null, 2)}`);
    }
    
    /**
     * Update workflow state from API response
     * @param {Object} workflowData - Complete workflow state from API
     */
    updateWorkflowState(workflowData) {
        console.log('[WorkflowVisualizer] Entry: updateWorkflowState()', workflowData);
        
        this.workflowData = workflowData;
        this.correlationId = workflowData.correlation_id;
        
        // Update node states based on workflow data
        if (workflowData.steps) {
            workflowData.steps.forEach(step => {
                const node = this.nodes.find(n => n.id === step.step_id);
                if (node) {
                    node.state = step.status; // 'pending', 'processing', 'success', 'error'
                    node.data = step.data;
                    
                    if (step.status === 'processing') {
                        this.currentStep = step.step_id;
                    }
                }
            });
            
            // Re-render nodes to update colors
            this.updateNodeStates();
        }
        
        // Update status bar
        this.updateStatusBar();
        
        console.log('[WorkflowVisualizer] Exit: updateWorkflowState()');
    }
    
    /**
     * Update node visual states
     * @private
     */
    updateNodeStates() {
        this.nodeGroup.selectAll('.node-circle')
            .transition()
            .duration(this.config.animationDuration)
            .attr('fill', d => this.getNodeColor(d));
    }
    
    /**
     * Update workflow status bar
     * @private
     */
    updateStatusBar() {
        const completedSteps = this.nodes.filter(n => n.state === 'success').length;
        const totalSteps = this.nodes.length;
        
        // Update UI elements
        const currentStepEl = document.getElementById('currentStepName');
        const progressEl = document.getElementById('workflowProgress');
        const correlationEl = document.getElementById('correlationId');
        
        if (currentStepEl) {
            const currentNode = this.nodes.find(n => n.id === this.currentStep);
            currentStepEl.textContent = currentNode ? currentNode.label.replace('\n', ' ') : '—';
        }
        
        if (progressEl) {
            progressEl.textContent = `${completedSteps}/${totalSteps}`;
        }
        
        if (correlationEl && this.correlationId) {
            correlationEl.textContent = this.correlationId.substring(0, 8) + '...';
            correlationEl.title = this.correlationId;
        }
    }
    
    /**
     * Populate Oracle modal with decision data
     * @private
     */
    populateOracleModal(oracleData) {
        // Decision badge
        const decisionBadge = document.getElementById('decisionBadge');
        if (decisionBadge && oracleData.decision) {
            decisionBadge.textContent = oracleData.decision.decision;
            decisionBadge.className = `decision-badge ${oracleData.decision.decision.toLowerCase()}`;
        }
        
        // Separation score
        if (oracleData.rkcnn && oracleData.rkcnn.separation_score !== undefined) {
            const scoreFill = document.getElementById('separationScoreFill');
            const scoreValue = document.getElementById('separationScoreValue');
            const score = oracleData.rkcnn.separation_score;
            
            if (scoreFill) {
                scoreFill.style.width = `${score * 100}%`;
            }
            if (scoreValue) {
                scoreValue.textContent = score.toFixed(3);
            }
        }
        
        // kNN neighbors
        if (oracleData.knn && oracleData.knn.neighbors) {
            this.renderKNNNeighbors(oracleData.knn.neighbors);
        }
        
        // Reason codes
        if (oracleData.decision && oracleData.decision.reasoning) {
            this.renderReasonCodes(oracleData.decision.reasoning);
        }
    }
    
    /**
     * Render kNN neighbors grid
     * @private
     */
    renderKNNNeighbors(neighbors) {
        const container = document.getElementById('knnNeighbors');
        if (!container) return;
        
        container.innerHTML = neighbors.map(neighbor => `
            <div class="knn-card">
                <div class="knn-preview">🎬</div>
                <div class="knn-info">
                    <strong>${neighbor.motion_id}</strong>
                    <span>Distance: ${neighbor.distance.toFixed(3)}</span>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Render reason code tags
     * @private
     */
    renderReasonCodes(reasoning) {
        const container = document.getElementById('reasonCodes');
        if (!container) return;
        
        const codes = Array.isArray(reasoning) ? reasoning : [reasoning];
        container.innerHTML = codes.map(code => `
            <span class="reason-tag">${code}</span>
        `).join('');
    }
    
    /**
     * Populate routing modal with transaction data
     * @private
     */
    populateRoutingModal(routingData) {
        // Route badge
        const routeBadge = document.getElementById('routeBadge');
        if (routeBadge && routingData.route) {
            routeBadge.textContent = routingData.route.toUpperCase();
            routeBadge.className = `route-badge ${routingData.route}`;
        }
        
        // Rationale
        const rationale = document.getElementById('routeRationale');
        if (rationale && routingData.rationale) {
            rationale.textContent = routingData.rationale;
        }
        
        // Render Sankey diagram
        if (routingData.operations) {
            this.renderSankeyDiagram(routingData);
        }
    }
    
    /**
     * Render Sankey diagram for transaction routing
     * @private
     */
    renderSankeyDiagram(routingData) {
        // TODO: Implement D3 Sankey diagram
        console.log('[WorkflowVisualizer] Rendering Sankey diagram', routingData);
    }
    
    /**
     * Setup event listeners for UI controls
     * @private
     */
    setupEventListeners() {
        // Reset workflow button
        const resetBtn = document.getElementById('resetWorkflowBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetWorkflow());
        }
        
        // Toggle layout button
        const toggleLayoutBtn = document.getElementById('toggleWorkflowLayoutBtn');
        if (toggleLayoutBtn) {
            toggleLayoutBtn.addEventListener('click', () => this.toggleLayout());
        }
        
        // Modal close buttons
        const closeOracleBtn = document.getElementById('closeOracleModal');
        if (closeOracleBtn) {
            closeOracleBtn.addEventListener('click', () => {
                document.getElementById('oracleModal').style.display = 'none';
            });
        }
        
        const closeRoutingBtn = document.getElementById('closeRoutingModal');
        if (closeRoutingBtn) {
            closeRoutingBtn.addEventListener('click', () => {
                document.getElementById('routingModal').style.display = 'none';
            });
        }
    }
    
    /**
     * Reset workflow to initial state
     */
    resetWorkflow() {
        console.log('[WorkflowVisualizer] Entry: resetWorkflow()');
        
        this.nodes.forEach(node => {
            node.state = 'pending';
            node.data = null;
        });
        
        this.currentStep = null;
        this.workflowData = null;
        this.updateNodeStates();
        this.updateStatusBar();
        
        console.log('[WorkflowVisualizer] Exit: resetWorkflow()');
    }
    
    /**
     * Toggle between vertical and horizontal layout
     */
    toggleLayout() {
        console.log('[WorkflowVisualizer] Toggling layout');
        this.config.layoutType = this.config.layoutType === 'vertical' ? 'horizontal' : 'vertical';
        // Re-initialize positions and simulation
        this.setupSimulation();
    }
    
    /**
     * Destroy visualizer and cleanup resources
     */
    destroy() {
        console.log('[WorkflowVisualizer] Entry: destroy()');
        
        if (this.simulation) {
            this.simulation.stop();
        }
        
        if (this.svg) {
            this.svg.remove();
        }
        
        console.log('[WorkflowVisualizer] Exit: destroy()');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WorkflowVisualizer;
}
