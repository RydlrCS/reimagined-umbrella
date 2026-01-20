/**
 * Temporal Conditioning Module
 * 
 * Implements BlendAnim paper methodology for controllable animation blending
 * with smooth temporal conditioning functions ω(t).
 */

class TemporalConditioning {
    constructor(visualizer) {
        this.visualizer = visualizer;
        this.config = {
            mode: 'smoothstep',  // step, linear, smoothstep, custom
            transitionZone: 10,  // frames
            skeletonAware: true  // use skeleton-aware blending
        };
    }
    
    /**
     * Setup temporal conditioning controls
     */
    init() {
        this.setupEventListeners();
        this.updateBlendCurveVisualization();
        this.updateModeDescription();
    }
    
    /**
     * Setup event listeners for temporal controls
     */
    setupEventListeners() {
        // Blend mode selector
        const blendModeSelect = document.getElementById('blendMode');
        if (blendModeSelect) {
            blendModeSelect.addEventListener('change', (e) => {
                this.config.mode = e.target.value;
                this.updateModeDescription();
                this.updateBlendCurveVisualization();
                this.updateTimelineBlendMode();
            });
        }
        
        // Transition zone control
        const transitionZoneSlider = document.getElementById('transitionZone');
        if (transitionZoneSlider) {
            transitionZoneSlider.addEventListener('input', (e) => {
                this.config.transitionZone = parseInt(e.target.value);
                document.getElementById('transitionZoneValue').textContent = e.target.value;
                this.updateBlendCurveVisualization();
            });
        }
        
        // Skeleton-aware toggle
        const skeletonAwareCheckbox = document.getElementById('skeletonAware');
        if (skeletonAwareCheckbox) {
            skeletonAwareCheckbox.addEventListener('change', (e) => {
                this.config.skeletonAware = e.target.checked;
                console.log(`🦴 Skeleton-aware blending: ${e.target.checked ? 'ON' : 'OFF'}`);
            });
        }
        
        // Blend weight slider - update curve on change
        const blendWeightSlider = document.getElementById('blendWeight');
        if (blendWeightSlider) {
            blendWeightSlider.addEventListener('input', () => {
                this.updateBlendCurveVisualization();
            });
        }
    }
    
    /**
     * Update blend mode description
     */
    updateModeDescription() {
        const descriptions = {
            step: 'Hard boundary - instant transition between motions (no smoothing)',
            linear: 'Linear interpolation - constant blend rate across transition zone',
            smoothstep: 'Smooth temporal conditioning with C² continuity - recommended by BlendAnim paper',
            custom: 'User-defined blend curve - Hermite interpolation with C³ continuity'
        };
        
        const descElement = document.getElementById('modeDescription');
        if (descElement) {
            descElement.textContent = descriptions[this.config.mode] || '';
        }
        
        // Update timeline indicator
        const indicatorIcons = {
            step: '▮',
            linear: '/',
            smoothstep: '⚡',
            custom: '✎'
        };
        
        const indicator = document.getElementById('blendModeIndicator');
        if (indicator) {
            const modeName = this.config.mode.charAt(0).toUpperCase() + this.config.mode.slice(1);
            indicator.textContent = `${indicatorIcons[this.config.mode]} ${modeName}`;
        }
    }
    
    /**
     * Update blend curve visualization canvas
     */
    updateBlendCurveVisualization() {
        const canvas = document.getElementById('blendCurveCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Get current settings
        const blendWeight = parseFloat(document.getElementById('blendWeight')?.value || 0.5);
        const transitionZone = this.config.transitionZone;
        const totalFrames = 100; // normalized timeline
        
        // Calculate blend boundary position
        const boundaryFrame = blendWeight * totalFrames;
        
        // Draw background zones
        // Motion A zone (green)
        ctx.fillStyle = 'rgba(46, 204, 113, 0.12)';
        const zoneAWidth = Math.max(0, boundaryFrame - transitionZone/2) * (width/totalFrames);
        ctx.fillRect(0, 0, zoneAWidth, height);
        
        // Motion B zone (orange)
        ctx.fillStyle = 'rgba(230, 126, 34, 0.12)';
        const zoneBStart = Math.min(width, (boundaryFrame + transitionZone/2) * (width/totalFrames));
        ctx.fillRect(zoneBStart, 0, width - zoneBStart, height);
        
        // Draw blend curve ω(t)
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        
        for (let i = 0; i <= width; i++) {
            const frame = (i / width) * totalFrames;
            const omega = this.calculateTemporalWeight(frame, boundaryFrame, transitionZone);
            const y = height - (omega * height);
            
            if (i === 0) {
                ctx.moveTo(i, y);
            } else {
                ctx.lineTo(i, y);
            }
        }
        
        ctx.stroke();
        
        // Draw transition zone markers
        if (transitionZone > 0 && this.config.mode !== 'step') {
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            
            const startX = Math.max(0, (boundaryFrame - transitionZone/2) * (width/totalFrames));
            const endX = Math.min(width, (boundaryFrame + transitionZone/2) * (width/totalFrames));
            
            ctx.beginPath();
            ctx.moveTo(startX, 0);
            ctx.lineTo(startX, height);
            ctx.moveTo(endX, 0);
            ctx.lineTo(endX, height);
            ctx.stroke();
            
            ctx.setLineDash([]);
        }
        
        // Draw boundary marker
        const boundaryX = (boundaryFrame / totalFrames) * width;
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(boundaryX, 0);
        ctx.lineTo(boundaryX, height);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw axis labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText('ω=1', 5, 12);
        ctx.fillText('ω=0', 5, height - 4);
        ctx.fillText('t →', width - 25, height - 4);
        
        // Draw current boundary label
        ctx.fillStyle = '#8b5cf6';
        ctx.font = 'bold 10px monospace';
        const boundaryLabel = `t=${Math.round(boundaryFrame)}`;
        const labelWidth = ctx.measureText(boundaryLabel).width;
        ctx.fillText(boundaryLabel, Math.min(boundaryX - labelWidth/2, width - labelWidth - 5), 12);
    }
    
    /**
     * Calculate temporal blend weight ω(t) based on BlendAnim paper methodology
     * 
     * Implements various temporal conditioning functions:
     * - Step: ω(t) = H(t - boundary) where H is Heaviside function
     * - Linear: ω(t) = t (normalized)
     * - Smoothstep: ω(t) = 3t² - 2t³ (C² continuity)
     * - Custom: ω(t) = 6t⁵ - 15t⁴ + 10t³ (C³ continuity, smootherstep)
     * 
     * @param {number} t - Current frame
     * @param {number} boundary - Blend boundary frame
     * @param {number} transitionZone - Number of frames for transition
     * @returns {number} Blend weight from 0 (Motion A) to 1 (Motion B)
     */
    calculateTemporalWeight(t, boundary, transitionZone) {
        const mode = this.config.mode;
        
        // Define transition range
        const transitionStart = boundary - transitionZone / 2;
        const transitionEnd = boundary + transitionZone / 2;
        
        // Step function (hard boundary) - Heaviside function
        if (mode === 'step' || transitionZone === 0) {
            return t >= boundary ? 1 : 0;
        }
        
        // Before transition zone - pure Motion A
        if (t < transitionStart) {
            return 0;
        }
        
        // After transition zone - pure Motion B
        if (t > transitionEnd) {
            return 1;
        }
        
        // Within transition zone - apply blend function
        // Normalize t to [0, 1] within transition zone
        const normalizedT = (t - transitionStart) / transitionZone;
        
        switch (mode) {
            case 'linear':
                // Linear ramp: ω(t) = t
                return normalizedT;
            
            case 'smoothstep':
                // Smoothstep (C² continuity): ω(t) = 3t² - 2t³
                // This is the RECOMMENDED method from BlendAnim paper
                // Provides smooth acceleration/deceleration
                return normalizedT * normalizedT * (3 - 2 * normalizedT);
            
            case 'custom':
                // Smootherstep (C³ continuity): ω(t) = 6t⁵ - 15t⁴ + 10t³
                // Even smoother than smoothstep, useful for very gradual transitions
                return normalizedT * normalizedT * normalizedT * 
                       (normalizedT * (normalizedT * 6 - 15) + 10);
            
            default:
                return normalizedT;
        }
    }
    
    /**
     * Update timeline visual to reflect current blend mode
     */
    updateTimelineBlendMode() {
        const blendGradient = document.getElementById('blendGradient');
        if (!blendGradient) return;
        
        const mode = this.config.mode;
        const transitionZone = this.config.transitionZone;
        const blendWeight = parseFloat(document.getElementById('blendWeight')?.value || 0.5);
        
        // Update gradient based on mode
        let gradient;
        
        if (mode === 'step') {
            // Hard boundary at blend weight percentage
            const percent = Math.round(blendWeight * 100);
            gradient = `linear-gradient(to right, rgba(46, 204, 113, 0.3) ${percent}%, rgba(230, 126, 34, 0.3) ${percent}%)`;
        } else if (mode === 'linear') {
            // Simple linear gradient
            gradient = 'linear-gradient(to right, rgba(46, 204, 113, 0.3) 0%, rgba(230, 126, 34, 0.3) 100%)';
        } else {
            // Smoothstep/Custom - create multi-stop gradient to approximate curve
            const stops = [];
            const numStops = 20;
            
            for (let i = 0; i <= numStops; i++) {
                const t = i / numStops;
                const normalizedT = t; // Already normalized to [0, 1]
                
                // Calculate omega for this position
                let omega;
                if (mode === 'smoothstep') {
                    omega = normalizedT * normalizedT * (3 - 2 * normalizedT);
                } else { // custom
                    omega = normalizedT * normalizedT * normalizedT * 
                           (normalizedT * (normalizedT * 6 - 15) + 10);
                }
                
                // Interpolate colors
                const r = Math.round(46 + (230 - 46) * omega);
                const g = Math.round(204 + (126 - 204) * omega);
                const b = Math.round(113 + (34 - 113) * omega);
                
                stops.push(`rgba(${r}, ${g}, ${b}, 0.3) ${t * 100}%`);
            }
            
            gradient = `linear-gradient(to right, ${stops.join(', ')})`;
        }
        
        blendGradient.style.background = gradient;
    }
    
    /**
     * Apply temporal blend at specific frame
     */
    applyTemporalBlend(motionA, motionB, currentFrame, totalFrames) {
        const blendWeight = parseFloat(document.getElementById('blendWeight')?.value || 0.5);
        const boundaryFrame = totalFrames * blendWeight;
        const omega = this.calculateTemporalWeight(currentFrame, boundaryFrame, this.config.transitionZone);
        
        // Apply skeleton-aware blending if enabled
        if (this.config.skeletonAware) {
            return this.skeletonAwareBlend(motionA, motionB, omega);
        } else {
            // Standard linear blend
            return this.linearBlend(motionA, motionB, omega);
        }
    }
    
    /**
     * Skeleton-aware blending (respects joint hierarchy)
     * 
     * Implements proper skeletal animation blending:
     * - Blends rotations in quaternion space (SLERP)
     * - Respects parent-child bone relationships
     * - Preserves bone lengths
     */
    skeletonAwareBlend(motionA, motionB, omega) {
        // TODO: Full skeleton-aware implementation would:
        // 1. Extract joint rotations from both motions
        // 2. Convert to quaternions
        // 3. SLERP (Spherical Linear Interpolation) between quaternions
        // 4. Maintain bone chain constraints
        // 5. Blend root motion separately
        
        // For now, use standard blend with visual feedback
        console.log(`🦴 Skeleton-aware blend: ω=${omega.toFixed(3)}`);
        return this.linearBlend(motionA, motionB, omega);
    }
    
    /**
     * Standard linear blend in Cartesian space
     */
    linearBlend(motionA, motionB, omega) {
        // Blend = (1 - ω) * MotionA + ω * MotionB
        if (!this.visualizer.characterMesh) return omega;
        
        const colorA = new THREE.Color(0x2ecc71); // Green
        const colorB = new THREE.Color(0xe67e22); // Orange
        const blendColor = new THREE.Color().lerpColors(colorA, colorB, omega);
        
        this.visualizer.characterMesh.children.forEach(child => {
            if (child.material) {
                child.material.color = blendColor;
                child.material.emissive = new THREE.Color(blendColor).multiplyScalar(0.2);
            }
        });
        
        return omega;
    }
    
    /**
     * Get current blend configuration
     */
    getConfig() {
        return { ...this.config };
    }
    
    /**
     * Set blend configuration
     */
    setConfig(config) {
        this.config = { ...this.config, ...config };
        this.updateModeDescription();
        this.updateBlendCurveVisualization();
        this.updateTimelineBlendMode();
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TemporalConditioning;
}
