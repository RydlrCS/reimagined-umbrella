/**
 * Metrics Dashboard Module
 * 
 * Visualizes blendanim quality metrics using Chart.js:
 * - Compact view with high-level summary cards
 * - Detailed modal view with full visualizations
 * - Quality tier gauge (Ultra/High/Medium/Low)
 * - Coverage line chart (30-frame windows)
 * - Diversity metrics (Local/Global bar chart)
 * - Joint smoothness heatmap (L2 velocity per joint)
 * - Cost breakdown pie chart
 * 
 * @module MetricsDashboard
 * @requires chart.js v4
 * @version 2.0.0
 */

class MetricsDashboard {
    /**
     * Initialize metrics dashboard
     * @param {string} containerId - DOM element ID for dashboard container
     * @param {Object} options - Configuration options
     */
    constructor(containerId = 'metricsDashboard', options = {}) {
        console.log(`[MetricsDashboard] Entry: constructor(containerId=${containerId})`);
        
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            updateInterval: options.updateInterval || 1000, // ms
            animationDuration: options.animationDuration || 500,
            colorScheme: options.colorScheme || 'default',
            ...options
        };
        
        // Chart instances
        this.charts = {
            qualityGauge: null,
            coverage: null,
            diversity: null,
            smoothness: null,
            cost: null
        };
        
        // Current metrics data
        this.metrics = null;
        
        // Quality tier thresholds (from blendanim specification)
        this.qualityTiers = {
            ultra: { coverage: 0.90, l2_velocity: 0.03, l2_acceleration: 0.015, smoothness: 0.94, cost: 0.25 },
            high: { coverage: 0.85, l2_velocity: 0.07, l2_acceleration: 0.04, smoothness: 0.86, cost: 0.10 },
            medium: { coverage: 0.75, l2_velocity: 0.10, l2_acceleration: 0.05, smoothness: 0.80, cost: 0.05 },
            low: { coverage: 0.0, l2_velocity: Infinity, l2_acceleration: Infinity, smoothness: 0.0, cost: 0.01 }
        };
        
        console.log('[MetricsDashboard] Exit: constructor()');
    }
    
    /**
     * Initialize dashboard and create all charts
     * @returns {Promise<void>}
     */
    async init() {
        console.log('[MetricsDashboard] Entry: init()');
        
        if (!this.container) {
            console.error(`[MetricsDashboard] Container #${this.containerId} not found`);
            return;
        }
        
        try {
            // Check if Chart.js is loaded
            if (typeof Chart === 'undefined') {
                throw new Error('Chart.js library not loaded');
            }
            
            // Create all charts
            this.createQualityGauge();
            this.createCoverageChart();
            this.createDiversityChart();
            this.createSmoothnessChart();
            this.createCostChart();
            
            // Setup event listeners
            this.setupEventListeners();
            
            console.log('[MetricsDashboard] Exit: init() - All charts initialized');
        } catch (error) {
            console.error('[MetricsDashboard] Error in init():', error);
            throw error;
        }
    }
    
    /**
     * Create quality tier gauge chart
     * @private
     */
    createQualityGauge() {
        console.log('[MetricsDashboard] Creating quality gauge chart');
        
        const canvas = document.getElementById('qualityGaugeChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.qualityGauge = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Ultra', 'High', 'Medium', 'Low', 'Remaining'],
                datasets: [{
                    data: [25, 25, 25, 25, 0], // Will be updated dynamically
                    backgroundColor: [
                        '#9b59b6', // Ultra - Purple
                        '#3498db', // High - Blue
                        '#f39c12', // Medium - Orange
                        '#95a5a6', // Low - Gray
                        '#ecf0f1'  // Remaining - Light gray
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                circumference: 180,
                rotation: 270,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Create coverage line chart
     * @private
     */
    createCoverageChart() {
        console.log('[MetricsDashboard] Creating coverage chart');
        
        const canvas = document.getElementById('coverageChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.coverage = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Frame windows
                datasets: [{
                    label: 'Coverage',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Threshold (Ultra)',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Coverage Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Frame Window'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    /**
     * Create diversity metrics bar chart
     * @private
     */
    createDiversityChart() {
        console.log('[MetricsDashboard] Creating diversity chart');
        
        const canvas = document.getElementById('diversityChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.diversity = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Local Diversity', 'Global Diversity'],
                datasets: [{
                    label: 'Diversity Score',
                    data: [0, 0],
                    backgroundColor: [
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(52, 152, 219, 0.8)'
                    ],
                    borderColor: [
                        '#2ecc71',
                        '#3498db'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Diversity Score'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    /**
     * Create joint smoothness heatmap chart
     * @private
     */
    createSmoothnessChart() {
        console.log('[MetricsDashboard] Creating smoothness heatmap');
        
        const canvas = document.getElementById('smoothnessChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Key joints to track
        const joints = ['Pelvis', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot'];
        
        this.charts.smoothness = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: joints,
                datasets: [{
                    label: 'L2 Velocity',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(231, 76, 60, 0.8)',
                    borderColor: '#e74c3c',
                    borderWidth: 1
                }, {
                    label: 'L2 Acceleration',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(230, 126, 34, 0.8)',
                    borderColor: '#e67e22',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y', // Horizontal bars
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'L2 Metric (lower = smoother)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    /**
     * Create cost breakdown pie chart
     * @private
     */
    createCostChart() {
        console.log('[MetricsDashboard] Creating cost breakdown chart');
        
        const canvas = document.getElementById('costChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        this.charts.cost = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Generation Cost', 'Quality Premium', 'Base Fee'],
                datasets: [{
                    data: [70, 20, 10],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(155, 89, 182, 0.8)',
                        'rgba(149, 165, 166, 0.8)'
                    ],
                    borderColor: '#fff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                return label + ': $' + value.toFixed(3) + ' USDC';
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update dashboard with new metrics data
     * @param {Object} metricsData - Metrics from blendanim service
     */
    updateMetrics(metricsData) {
        console.log('[MetricsDashboard] Entry: updateMetrics()', metricsData);
        
        this.metrics = metricsData;
        
        try {
            // Show the dashboard if hidden
            if (this.container) {
                const content = this.container.querySelector('.dashboard-content');
                if (content && content.style.display === 'none') {
                    content.style.display = 'block';
                    const toggleBtn = document.getElementById('toggleMetricsBtn');
                    if (toggleBtn) toggleBtn.textContent = '−';
                }
            }
            
            // Update summary cards
            this.updateSummaryCards(metricsData);
            
            // Update detail charts
            this.updateQualityGauge(metricsData);
            
            // Update coverage chart (handle both formats)
            if (metricsData.coverage_timeline) {
                this.updateCoverageChart(metricsData.coverage_timeline);
            } else if (metricsData.coverage !== undefined) {
                // Simple format: convert single value to timeline
                this.updateCoverageChart([metricsData.coverage]);
            }
            
            // Update diversity chart (handle both formats)
            if (metricsData.diversity) {
                this.updateDiversityChart(metricsData.diversity);
            } else if (metricsData.local_diversity !== undefined && metricsData.global_diversity !== undefined) {
                // Simple format: construct diversity object
                this.updateDiversityChart({
                    local: metricsData.local_diversity,
                    global: metricsData.global_diversity
                });
            }
            
            // Update smoothness chart (handle both formats)
            if (metricsData.per_joint_metrics) {
                this.updateSmoothnessChart(metricsData.per_joint_metrics);
            } else if (metricsData.l2_velocity !== undefined) {
                // Simple format: use average for all joints
                const avgVelocity = metricsData.l2_velocity;
                const jointMetrics = [];
                const jointNames = ['Hip', 'Spine', 'Shoulder_L', 'Shoulder_R', 'Elbow_L', 'Elbow_R', 'Knee_L', 'Knee_R'];
                jointNames.forEach(name => {
                    jointMetrics.push({
                        joint_name: name,
                        l2_velocity: avgVelocity * (0.8 + Math.random() * 0.4) // Add variation
                    });
                });
                this.updateSmoothnessChart(jointMetrics);
            }
            
            // Update cost chart
            if (metricsData.cost_breakdown) {
                this.updateCostChart(metricsData.cost_breakdown);
            }
            
            // Update tier label
            this.updateTierLabel(metricsData.quality_tier);
            
            console.log('[MetricsDashboard] Exit: updateMetrics() - All charts updated');
        } catch (error) {
            console.error('[MetricsDashboard] Error updating metrics:', error);
        }
    }
    
    /**
     * Update summary cards with high-level metrics
     * @private
     */
    updateSummaryCards(metrics) {
        // Quality tier
        const qualityEl = document.getElementById('qualityTierValue');
        if (qualityEl) {
            const tier = metrics.quality_tier || this.determineQualityTier(metrics);
            qualityEl.textContent = tier.toUpperCase();
            qualityEl.className = 'metric-value tier-' + tier;
        }
        
        // Coverage
        const coverageEl = document.getElementById('coverageValue');
        if (coverageEl) {
            const coverage = metrics.coverage || (metrics.coverage_timeline ? metrics.coverage_timeline[metrics.coverage_timeline.length - 1] : 0);
            coverageEl.textContent = `${(coverage * 100).toFixed(1)}%`;
        }
        
        // Diversity (average of local and global)
        const diversityEl = document.getElementById('diversityValue');
        if (diversityEl) {
            let diversity = 0;
            if (metrics.diversity) {
                diversity = (metrics.diversity.local + metrics.diversity.global) / 2;
            } else if (metrics.local_diversity !== undefined && metrics.global_diversity !== undefined) {
                diversity = (metrics.local_diversity + metrics.global_diversity) / 2;
            }
            diversityEl.textContent = `${(diversity * 100).toFixed(1)}%`;
        }
        
        // Smoothness (inverse of L2 velocity)
        const smoothnessEl = document.getElementById('smoothnessValue');
        if (smoothnessEl) {
            const l2vel = metrics.l2_velocity || 0.06;
            const smoothness = Math.max(0, 1 - (l2vel / 0.15)); // Normalize to 0-1
            smoothnessEl.textContent = `${(smoothness * 100).toFixed(1)}%`;
        }
        
        // Cost
        const costEl = document.getElementById('costValue');
        if (costEl) {
            let cost = '—';
            if (metrics.cost_breakdown && metrics.cost_breakdown.total) {
                cost = `$${metrics.cost_breakdown.total.toFixed(3)}`;
            } else if (metrics.quality_tier) {
                const tierCost = this.qualityTiers[metrics.quality_tier]?.cost || 0.05;
                cost = `$${tierCost.toFixed(3)}`;
            }
            costEl.textContent = cost;
        }
    }
    
    /**
     * Update quality gauge based on current metrics
     * @private
     */
    updateQualityGauge(metrics) {
        if (!this.charts.qualityGauge) return;
        
        const tier = this.determineQualityTier(metrics);
        const tierIndex = ['ultra', 'high', 'medium', 'low'].indexOf(tier);
        
        // Update gauge to show current tier
        const data = [0, 0, 0, 0, 100];
        if (tierIndex >= 0) {
            data[tierIndex] = 100;
            data[4] = 0;
        }
        
        this.charts.qualityGauge.data.datasets[0].data = data;
        this.charts.qualityGauge.update();
    }
    
    /**
     * Update coverage chart with timeline data
     * @private
     */
    updateCoverageChart(coverageTimeline) {
        if (!this.charts.coverage) return;
        
        const labels = coverageTimeline.map((_, i) => `W${i + 1}`);
        const threshold = Array(coverageTimeline.length).fill(this.qualityTiers.ultra.coverage);
        
        this.charts.coverage.data.labels = labels;
        this.charts.coverage.data.datasets[0].data = coverageTimeline;
        this.charts.coverage.data.datasets[1].data = threshold;
        this.charts.coverage.update();
    }
    
    /**
     * Update diversity chart
     * @private
     */
    updateDiversityChart(diversity) {
        if (!this.charts.diversity) return;
        
        this.charts.diversity.data.datasets[0].data = [
            diversity.local || 0,
            diversity.global || 0
        ];
        this.charts.diversity.update();
    }
    
    /**
     * Update smoothness heatmap
     * @private
     */
    updateSmoothnessChart(perJointMetrics) {
        if (!this.charts.smoothness) return;
        
        const joints = ['Pelvis', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot'];
        const velocities = joints.map(joint => perJointMetrics[joint]?.l2_velocity || 0);
        const accelerations = joints.map(joint => perJointMetrics[joint]?.l2_acceleration || 0);
        
        this.charts.smoothness.data.datasets[0].data = velocities;
        this.charts.smoothness.data.datasets[1].data = accelerations;
        this.charts.smoothness.update();
    }
    
    /**
     * Update cost breakdown chart
     * @private
     */
    updateCostChart(costBreakdown) {
        if (!this.charts.cost) return;
        
        this.charts.cost.data.datasets[0].data = [
            costBreakdown.generation_cost || 0,
            costBreakdown.quality_premium || 0,
            costBreakdown.base_fee || 0
        ];
        this.charts.cost.update();
        
        // Update total cost display
        const total = (costBreakdown.generation_cost || 0) + 
                     (costBreakdown.quality_premium || 0) + 
                     (costBreakdown.base_fee || 0);
        const costSummary = document.getElementById('costSummary');
        if (costSummary) {
            costSummary.textContent = `$${total.toFixed(3)} USDC`;
        }
    }
    
    /**
     * Update quality tier label
     * @private
     */
    updateTierLabel(tier) {
        const tierLabel = document.getElementById('tierLabel');
        if (tierLabel) {
            tierLabel.textContent = tier ? tier.toUpperCase() : '—';
            tierLabel.className = `tier-label tier-${tier}`;
        }
    }
    
    /**
     * Determine quality tier based on metrics
     * @private
     */
    determineQualityTier(metrics) {
        const coverage = metrics.coverage || 0;
        const l2_vel = metrics.l2_velocity || Infinity;
        const l2_acc = metrics.l2_acceleration || Infinity;
        const smoothness = metrics.blend_area_smoothness || 0;
        
        // Check ultra tier
        if (coverage >= this.qualityTiers.ultra.coverage &&
            l2_vel <= this.qualityTiers.ultra.l2_velocity &&
            l2_acc <= this.qualityTiers.ultra.l2_acceleration &&
            smoothness >= this.qualityTiers.ultra.smoothness) {
            return 'ultra';
        }
        
        // Check high tier
        if (coverage >= this.qualityTiers.high.coverage &&
            l2_vel <= this.qualityTiers.high.l2_velocity &&
            l2_acc <= this.qualityTiers.high.l2_acceleration &&
            smoothness >= this.qualityTiers.high.smoothness) {
            return 'high';
        }
        
        // Check medium tier
        if (coverage >= this.qualityTiers.medium.coverage &&
            l2_vel <= this.qualityTiers.medium.l2_velocity &&
            l2_acc <= this.qualityTiers.medium.l2_acceleration &&
            smoothness >= this.qualityTiers.medium.smoothness) {
            return 'medium';
        }
        
        return 'low';
    }
    
    /**
     * Setup event listeners
     * @private
     */
    setupEventListeners() {
        // Toggle metrics dashboard
        const toggleBtn = document.getElementById('toggleMetricsBtn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const content = this.container.querySelector('.dashboard-content');
                if (content) {
                    const isHidden = content.style.display === 'none';
                    content.style.display = isHidden ? 'block' : 'none';
                    toggleBtn.textContent = isHidden ? '−' : '+';
                }
            });
        }
        
        // View all metrics button - opens modal
        const viewAllBtn = document.getElementById('viewAllMetricsBtn');
        const metricsModal = document.getElementById('metricsModal');
        const closeModalBtn = document.getElementById('closeMetricsModal');
        
        if (viewAllBtn && metricsModal) {
            viewAllBtn.addEventListener('click', () => {
                metricsModal.style.display = 'flex';
                // Refresh charts when modal opens
                Object.values(this.charts).forEach(chart => {
                    if (chart) chart.resize();
                });
            });
        }
        
        if (closeModalBtn && metricsModal) {
            closeModalBtn.addEventListener('click', () => {
                metricsModal.style.display = 'none';
            });
        }
        
        // Click summary cards to open modal with specific section
        document.querySelectorAll('.metric-summary-card').forEach(card => {
            card.addEventListener('click', () => {
                if (metricsModal) {
                    metricsModal.style.display = 'flex';
                    // Could scroll to specific section based on data-metric attribute
                }
            });
        });
        
        // Close modal when clicking outside
        if (metricsModal) {
            metricsModal.addEventListener('click', (e) => {
                if (e.target === metricsModal) {
                    metricsModal.style.display = 'none';
                }
            });
        }
    }
    
    /**
     * Clear all charts
     */
    clear() {
        console.log('[MetricsDashboard] Clearing all charts');
        
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.data) {
                chart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                chart.update();
            }
        });
        
        this.metrics = null;
    }
    
    /**
     * Destroy dashboard and cleanup resources
     */
    destroy() {
        console.log('[MetricsDashboard] Entry: destroy()');
        
        // Destroy all Chart.js instances
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        
        this.charts = {};
        this.metrics = null;
        
        console.log('[MetricsDashboard] Exit: destroy()');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsDashboard;
}
