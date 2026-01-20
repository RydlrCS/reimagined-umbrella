/**
 * Kinetic Ledger Motion Visualizer
 * 
 * Three.js-based 3D visualizer for motion blending with Web3 wallet integration.
 * Demonstrates Capoeira to Breakdance motion blend using Mixamo animations.
 */

class MotionVisualizer {
    constructor() {
        // Three.js scene components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.characterMesh = null;
        this.npcs = [];
        this.gridHelper = null;
        
        // Animation state
        this.clock = new THREE.Clock();
        this.mixer = null;
        this.currentAnimation = null;
        this.animations = {};
        this.isPlaying = false;
        this.currentTime = 0;
        
        // Motion library
        this.motionLibrary = [];
        this.selectedMotions = { A: null, B: null };
        this.currentBlend = null;
        
        // Wallet state
        this.wallet = {
            connected: false,
            address: null,
            balance: 0,
            provider: null
        };
        
        // UI state
        this.stats = {
            motionCount: 0,
            npcCount: 0,
            blendCount: 0
        };
        
        // NPC spawning state
        this.autoSpawnInterval = null;
        this.autoPaymentEnabled = false;
        
        // Temporal conditioning module (Paper enhancement)
        this.temporalConditioning = null;
        
        // Transaction history
        this.transactions = [];
        
        // FPS tracking
        this.fpsHistory = [];
        this.lastFrameTime = performance.now();
        
        // Config
        this.config = {
            USDC_ADDRESS: '0x036CbD53842c5426634e7929541eC2318f3dCF7e', // Arc testnet USDC
            PLATFORM_ADDRESS: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1', // Platform wallet
            API_BASE_URL: window.location.origin,
        };
    }
    
    /**
     * Initialize the visualizer
     */
    async init() {
        console.log('[MotionVisualizer] Entry: init()');
        console.log('🚀 Initializing Kinetic Ledger Visualizer...');
        
        try {
            // Apply feature flags first
            this.applyFeatureFlags();
            
            // Setup Three.js scene
            this.setupScene();
            
            // Initialize workflow visualizer
            console.log('[MotionVisualizer] Initializing workflow visualizer...');
            if (typeof WorkflowVisualizer !== 'undefined') {
                window.workflowVisualizer = new WorkflowVisualizer();
                await window.workflowVisualizer.init();
                console.log('[MotionVisualizer] Workflow visualizer initialized');
            }
            
            // Initialize metrics dashboard
            console.log('[MotionVisualizer] Initializing metrics dashboard...');
            if (typeof MetricsDashboard !== 'undefined') {
                window.metricsDashboard = new MetricsDashboard();
                await window.metricsDashboard.init();
                console.log('[MotionVisualizer] Metrics dashboard initialized');
            }
            
            // Load motion library
            await this.loadMotionLibrary();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize temporal conditioning module (Paper enhancement)
            if (typeof TemporalConditioning !== 'undefined') {
                this.temporalConditioning = new TemporalConditioning(this);
                this.temporalConditioning.init();
                console.log('[MotionVisualizer] ⚡ Temporal conditioning initialized (BlendAnim paper method)');
            }
            
            // Setup video modal listeners
            this.setupVideoModalListeners();
            
            // Start animation loop
            this.animate();
            
            // Auto-load demo blend (Capoeira to Breakdance) if enabled
            if (window.FEATURE_FLAGS?.AUTO_LOAD_DEMO !== false) {
                await this.loadDemoBlend();
            }
            
            console.log('[MotionVisualizer] Exit: init()');
            console.log('✅ Visualizer initialized successfully');
        } catch (error) {
            console.error('[MotionVisualizer] Error in init():', error);
            console.error('❌ Failed to initialize visualizer:', error);
            this.showError('Failed to initialize visualizer: ' + error.message);
        }
    }
    
    /**
     * Setup Three.js scene
     */
    setupScene() {
        const canvas = document.getElementById('threeCanvas');
        const container = canvas.parentElement;
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(5, 3, 5);
        this.camera.lookAt(0, 1, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ 
            canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.set(0, 1, 0);
        this.controls.update();
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        this.scene.add(directionalLight);
        
        // Add grid helper
        this.gridHelper = new THREE.GridHelper(20, 20, 0x475569, 0x334155);
        this.scene.add(this.gridHelper);
        
        // Add character placeholder (will be replaced with FBX model)
        this.createCharacterPlaceholder();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        console.log('✅ Three.js scene created');
    }
    
    /**
     * Create character placeholder mesh
     */
    createCharacterPlaceholder() {
        // Create simple humanoid character using boxes
        const group = new THREE.Group();
        
        // Body (torso)
        const bodyGeometry = new THREE.BoxGeometry(0.8, 1.2, 0.4);
        const bodyMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x6366f1,
            emissive: 0x1e1b4b,
            shininess: 30
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 1.5;
        body.castShadow = true;
        group.add(body);
        
        // Head
        const headGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 2.4;
        head.castShadow = true;
        group.add(head);
        
        // Arms
        const armGeometry = new THREE.BoxGeometry(0.2, 1.0, 0.2);
        const leftArm = new THREE.Mesh(armGeometry, bodyMaterial);
        leftArm.position.set(-0.6, 1.5, 0);
        leftArm.castShadow = true;
        group.add(leftArm);
        
        const rightArm = new THREE.Mesh(armGeometry, bodyMaterial);
        rightArm.position.set(0.6, 1.5, 0);
        rightArm.castShadow = true;
        group.add(rightArm);
        
        // Legs
        const legGeometry = new THREE.BoxGeometry(0.25, 1.2, 0.25);
        const leftLeg = new THREE.Mesh(legGeometry, bodyMaterial);
        leftLeg.position.set(-0.25, 0.5, 0);
        leftLeg.castShadow = true;
        group.add(leftLeg);
        
        const rightLeg = new THREE.Mesh(legGeometry, bodyMaterial);
        rightLeg.position.set(0.25, 0.5, 0);
        rightLeg.castShadow = true;
        group.add(rightLeg);
        
        this.characterMesh = group;
        this.scene.add(group);
        
        // Add simple idle animation (bobbing)
        this.addIdleAnimation(group);
    }
    
    /**
     * Add simple idle animation to character
     */
    addIdleAnimation(character) {
        const idleAnimation = () => {
            if (!this.isPlaying && character === this.characterMesh) {
                const time = Date.now() * 0.001;
                character.position.y = Math.sin(time * 2) * 0.05;
                character.rotation.y = Math.sin(time * 0.5) * 0.05;
            }
            requestAnimationFrame(idleAnimation);
        };
        idleAnimation();
    }
    
    /**
     * Load motion library from backend
     */
    async loadMotionLibrary() {
        try {
            const response = await fetch(`${this.config.API_BASE_URL}/api/motions/library`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.motionLibrary = data.motions || [];
            
            this.stats.motionCount = this.motionLibrary.length;
            this.updateStats();
            
            // Render motion cards
            this.renderMotionCards();
            
            console.log(`✅ Loaded ${this.motionLibrary.length} motions`);
        } catch (error) {
            console.warn('⚠️ Failed to load motion library from backend, using sample data:', error);
            
            // Use sample data for demo
            this.motionLibrary = [
                {
                    id: 'capoeira',
                    name: 'Capoeira',
                    tags: ['dance', 'combat', 'acrobatic'],
                    duration: 4.5,
                    novelty: 0.75,
                    filepath: 'data/mixamo_anims/fbx/X Bot@Capoeira.fbx'
                },
                {
                    id: 'breakdance',
                    name: 'Breakdance Freeze',
                    tags: ['dance', 'urban', 'freeze'],
                    duration: 3.8,
                    novelty: 0.82,
                    filepath: 'data/mixamo_anims/fbx/X Bot@Breakdance Freeze Var 2.fbx'
                }
            ];
            
            this.stats.motionCount = this.motionLibrary.length;
            this.updateStats();
            this.renderMotionCards();
        }
    }
    
    /**
     * Render motion cards in sidebar
     */
    renderMotionCards() {
        const motionList = document.getElementById('motionList');
        motionList.innerHTML = '';
        
        this.motionLibrary.forEach(motion => {
            const card = document.createElement('div');
            card.className = 'motion-card';
            card.dataset.motionId = motion.id;
            
            card.innerHTML = `
                <div class="motion-name">${motion.name}</div>
                <div class="motion-tags">
                    ${motion.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <div class="motion-meta">
                    <span>Duration: ${motion.duration}s</span>
                    <span>Novelty: ${(motion.novelty * 100).toFixed(0)}%</span>
                </div>
            `;
            
            card.addEventListener('click', () => this.selectMotion(motion));
            motionList.appendChild(card);
        });
    }
    
    /**
     * Select motion for blending
     */
    selectMotion(motion) {
        // Determine which slot to use (A or B)
        const slot = this.selectedMotions.A === null ? 'A' : 'B';
        this.selectedMotions[slot] = motion;
        
        // Update UI
        document.getElementById(`motion${slot}`).textContent = motion.name;
        
        // Highlight selected card
        document.querySelectorAll('.motion-card').forEach(card => {
            if (card.dataset.motionId === motion.id) {
                card.classList.add('selected');
            }
        });
        
        // Enable blend button if both motions selected
        const applyBlendBtn = document.getElementById('applyBlendBtn');
        if (this.selectedMotions.A && this.selectedMotions.B) {
            applyBlendBtn.disabled = false;
        }
        
        console.log(`Selected ${motion.name} for slot ${slot}`);
    }
    
    /**
     * Load demo blend (Capoeira to Breakdance)
     */
    async loadDemoBlend() {
        console.log('🎭 Loading demo blend: Capoeira → Breakdance');
        
        // Auto-select both motions
        const capoeira = this.motionLibrary.find(m => m.id === 'capoeira');
        const breakdance = this.motionLibrary.find(m => m.id === 'breakdance');
        
        if (capoeira && breakdance) {
            this.selectMotion(capoeira);
            setTimeout(() => this.selectMotion(breakdance), 100);
            
            // Auto-apply blend after 500ms
            setTimeout(() => {
                const weight = 0.5; // 50/50 blend
                this.applyManualBlend(weight);
                
                // Update blend weight slider
                const blendWeightSlider = document.getElementById('blendWeight');
                blendWeightSlider.value = weight;
                document.getElementById('blendWeightValue').textContent = weight.toFixed(2);
            }, 500);
        }
    }
    
    /**
     * Apply manual blend
     */
    applyManualBlend(weight) {
        if (!this.selectedMotions.A || !this.selectedMotions.B) {
            this.showError('Please select two motions to blend');
            return;
        }
        
        const motionA = this.selectedMotions.A;
        const motionB = this.selectedMotions.B;
        
        console.log(`🎨 Blending: ${motionA.name} (${(1-weight)*100}%) + ${motionB.name} (${weight*100}%)`);
        
        // Create blend visualization
        this.visualizeBlend(motionA, motionB, weight);
        
        // Update timeline
        this.updateTimelineSegments([
            { name: motionA.name, weight: 1 - weight, color: '#6366f1' },
            { name: motionB.name, weight: weight, color: '#8b5cf6' }
        ]);
        
        // Update stats
        this.stats.blendCount++;
        this.updateStats();
        
        // Show success message
        this.showSuccess(`Blend applied: ${motionA.name} → ${motionB.name}`);
    }
    
    /**
     * Visualize blend on character
     */
    visualizeBlend(motionA, motionB, weight) {
        // Animate character to show blend
        const character = this.characterMesh;
        if (!character) return;
        
        // Change color based on blend weight
        const colorA = new THREE.Color(0x6366f1); // Primary color
        const colorB = new THREE.Color(0x8b5cf6); // Secondary color
        const blendColor = new THREE.Color().lerpColors(colorA, colorB, weight);
        
        character.children.forEach(child => {
            if (child.material) {
                child.material.color = blendColor;
                child.material.emissive = new THREE.Color(blendColor).multiplyScalar(0.3);
            }
        });
        
        // Animate blend transition
        this.animateBlendTransition(motionA, motionB, weight);
        
        // Store current blend
        this.currentBlend = {
            motionA,
            motionB,
            weight,
            timestamp: Date.now()
        };
    }
    
    /**
     * Animate blend transition
     */
    animateBlendTransition(motionA, motionB, weight) {
        const character = this.characterMesh;
        const duration = 2000; // 2 seconds
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const eased = progress < 0.5
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;
            
            // Animate character rotation and position
            character.rotation.y = Math.sin(eased * Math.PI * 4) * 0.5;
            character.position.y = Math.abs(Math.sin(eased * Math.PI * 2)) * 0.5;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                // Reset to neutral position
                character.rotation.y = 0;
                character.position.y = 0;
            }
        };
        
        animate();
    }
    
    /**
     * Update timeline segments
     */
    updateTimelineSegments(segments) {
        // Update the new frame-by-frame timeline
        this.updateFrameTimeline(segments);
    }
    
    /**
     * Update frame-by-frame timeline with pose visualization
     */
    updateFrameTimeline(segments) {
        console.log('[MotionVisualizer] Updating frame timeline', segments);
        
        // Update labels
        if (segments[0]) {
            const motionALabel = document.getElementById('motionALabel');
            if (motionALabel) motionALabel.textContent = segments[0].name || 'Motion A';
        }
        
        if (segments[1]) {
            const motionBLabel = document.getElementById('motionBLabel');
            if (motionBLabel) motionBLabel.textContent = segments[1].name || 'Motion B';
            
            const blendLabel = document.getElementById('blendLabel');
            if (blendLabel) {
                blendLabel.textContent = `${segments[0]?.name || 'Motion A'}-to-${segments[1]?.name || 'Motion B'}`;
            }
        }
        
        // Update timeline info
        const timelineInfo = document.getElementById('timelineInfo');
        if (timelineInfo) {
            timelineInfo.textContent = `${segments.length} motions loaded`;
        }
        
        // Generate pose silhouettes for blend
        this.generatePoseSilhouettes();
    }
    
    /**
     * Generate pose silhouettes for the blend timeline
     */
    generatePoseSilhouettes() {
        const poseSilhouettesContainer = document.getElementById('poseSilhouettes');
        if (!poseSilhouettesContainer) return;
        
        poseSilhouettesContainer.innerHTML = '';
        
        // Generate 40 pose frames
        const numFrames = 40;
        for (let i = 0; i < numFrames; i++) {
            const frameDiv = document.createElement('div');
            frameDiv.className = 'pose-frame';
            
            const silhouette = document.createElement('div');
            silhouette.className = 'pose-silhouette';
            
            // Vary pose heights to simulate different poses
            const progress = i / numFrames;
            
            // Create transition effect: start standing, middle transition, end different pose
            if (progress < 0.3) {
                // Start - stable poses
                silhouette.classList.add('pose-standing');
            } else if (progress < 0.7) {
                // Middle - transitioning
                silhouette.classList.add('pose-transitioning');
                // Add some variation
                if (i % 3 === 0) silhouette.classList.add('pose-crouching');
            } else {
                // End - different stable pose
                silhouette.classList.add(i % 2 === 0 ? 'pose-jumping' : 'pose-standing');
            }
            
            frameDiv.appendChild(silhouette);
            poseSilhouettesContainer.appendChild(frameDiv);
        }
        
        console.log('[MotionVisualizer] Generated', numFrames, 'pose silhouettes');
    }
    
    /**
     * Update timeline with blend frame overlays
     */
    updateTimelineWithBlendFrames(segments, blendFrames) {
        console.log('[MotionVisualizer] Entry: updateTimelineWithBlendFrames', segments, blendFrames);
        
        const container = document.getElementById('blendSegments');
        container.innerHTML = '';
        
        const totalDuration = segments.reduce((sum, s) => sum + (s.duration || 5), 0);
        let currentTime = 0;
        
        segments.forEach((segment, index) => {
            const div = document.createElement('div');
            div.className = 'blend-segment';
            div.style.flex = segment.weight;
            div.style.background = `linear-gradient(135deg, ${segment.color}50, ${segment.color}80)`;
            div.style.borderColor = segment.color;
            
            // Calculate segment duration in frames (assuming 30 fps)
            const segmentDurationFrames = (segment.duration || 5) * 30;
            const segmentStartFrame = currentTime * 30;
            const segmentEndFrame = segmentStartFrame + segmentDurationFrames;
            
            // Create blend frame overlay if this segment contains blend frames
            let overlayHTML = '';
            if (blendFrames && blendFrames.start !== undefined && blendFrames.end !== undefined) {
                const blendStart = blendFrames.start;
                const blendEnd = blendFrames.end;
                
                // Check if blend frames overlap with this segment
                if (blendStart < segmentEndFrame && blendEnd > segmentStartFrame) {
                    // Calculate overlay position and width
                    const overlapStart = Math.max(blendStart, segmentStartFrame);
                    const overlapEnd = Math.min(blendEnd, segmentEndFrame);
                    
                    const overlayLeft = ((overlapStart - segmentStartFrame) / segmentDurationFrames) * 100;
                    const overlayWidth = ((overlapEnd - overlapStart) / segmentDurationFrames) * 100;
                    
                    overlayHTML = `
                        <div class="blend-frame-overlay blend-active" 
                             style="left: ${overlayLeft}%; width: ${overlayWidth}%;"
                             title="Blend frames: ${blendStart}-${blendEnd}">
                        </div>
                    `;
                }
            }
            
            div.innerHTML = `
                <div class="segment-label">${segment.name}</div>
                <div class="segment-info">${(segment.weight * 100).toFixed(0)}% • ${segment.duration}s</div>
                <div class="segment-info" style="font-size: 0.6rem;">Frames: ${segmentStartFrame.toFixed(0)}-${segmentEndFrame.toFixed(0)}</div>
                ${overlayHTML}
            `;
            
            container.appendChild(div);
            currentTime += segment.duration || 5;
        });
        
        console.log('[MotionVisualizer] Exit: updateTimelineWithBlendFrames - Timeline updated with frame overlays');
    }
    
    /**
     * Load metrics to dashboard
     */
    loadMetricsToDashboard(metrics) {
        console.log('[MotionVisualizer] Entry: loadMetricsToDashboard', metrics);
        
        // Initialize metrics dashboard if not already done
        if (!window.metricsDashboard) {
            window.metricsDashboard = new MetricsDashboard();
            window.metricsDashboard.init().catch(err => {
                console.error('[MotionVisualizer] Failed to initialize metrics dashboard:', err);
            });
        }
        
        // Update dashboard with metrics
        if (window.metricsDashboard && window.metricsDashboard.updateMetrics) {
            window.metricsDashboard.updateMetrics(metrics);
            console.log('[MotionVisualizer] Metrics loaded to dashboard');
        }
        
        console.log('[MotionVisualizer] Exit: loadMetricsToDashboard');
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Wallet connection
        document.getElementById('connectWalletBtn').addEventListener('click', () => this.connectWallet());
        document.getElementById('disconnectWalletBtn').addEventListener('click', () => this.disconnectWallet());
        
        // Blend controls
        document.getElementById('blendWeight').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('blendWeightValue').textContent = value.toFixed(2);
        });
        
        document.getElementById('transitionSpeed').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('transitionSpeedValue').textContent = `${value.toFixed(1)}x`;
        });
        
        document.getElementById('applyBlendBtn').addEventListener('click', () => {
            const weight = parseFloat(document.getElementById('blendWeight').value);
            this.applyManualBlend(weight);
        });
        
        // Prompt-based generation
        document.getElementById('generateBlendBtn').addEventListener('click', () => this.generateBlendFromPrompt());
        
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                const prompt = e.target.dataset.prompt;
                document.getElementById('blendPrompt').value = prompt;
                this.updateCostEstimate();
            });
        });
        
        document.getElementById('blendPrompt').addEventListener('input', () => this.updateCostEstimate());
        document.getElementById('blendQuality').addEventListener('change', () => this.updateCostEstimate());
        
        // NPC spawning
        document.getElementById('spawnNPCBtn').addEventListener('click', () => this.spawnNPC());
        document.getElementById('autoSpawn').addEventListener('change', (e) => this.toggleAutoSpawn(e.target.checked));
        document.getElementById('autoPayment').addEventListener('change', (e) => {
            this.autoPaymentEnabled = e.target.checked;
        });
        
        document.getElementById('energyLevel').addEventListener('input', (e) => {
            document.getElementById('energyLevelValue').textContent = e.target.value;
        });
        
        // Playback controls
        document.getElementById('playBtn').addEventListener('click', () => this.play());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
        document.getElementById('stopBtn').addEventListener('click', () => this.stop());
        
        // Camera controls
        document.getElementById('resetCamera').addEventListener('click', () => this.resetCamera());
        document.getElementById('toggleGrid').addEventListener('click', () => this.toggleGrid());
        
        // Export
        document.getElementById('exportBlendBtn').addEventListener('click', () => this.exportBlend());
        document.getElementById('mintNFTBtn').addEventListener('click', () => this.mintNFT());
        
        // Search and filters
        document.getElementById('searchMotions').addEventListener('input', (e) => this.filterMotions(e.target.value));
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyFilter(e.target.dataset.filter));
        });
    }
    
    /**
     * Setup video modal event listeners
     */
    setupVideoModalListeners() {
        // Close modal button
        const closeBtn = document.getElementById('closeVideoModal');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeVideoModal());
        }
        
        // Modal overlay click
        const overlay = document.getElementById('videoModalOverlay');
        if (overlay) {
            overlay.addEventListener('click', () => this.closeVideoModal());
        }
        
        // View blend button
        const viewBlendBtn = document.getElementById('viewBlendBtn');
        if (viewBlendBtn) {
            viewBlendBtn.addEventListener('click', () => {
                if (this.currentBlend && this.currentBlend.video) {
                    this.showVideoModal(this.currentBlend.video);
                }
            });
        }
        
        // Download video button
        const downloadBtn = document.getElementById('downloadVideoBtn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                if (this.currentBlend && this.currentBlend.video && this.currentBlend.video.video_uri) {
                    window.open(this.currentBlend.video.video_uri, '_blank');
                }
            });
        }
        
        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeVideoModal();
            }
        });
    }
    
    /**
     * Connect Web3 wallet
     */
    async connectWallet() {
        try {
            if (typeof window.ethereum === 'undefined') {
                throw new Error('MetaMask not installed. Please install MetaMask to continue.');
            }
            
            console.log('🔌 Connecting wallet...');
            
            const accounts = await window.ethereum.request({ 
                method: 'eth_requestAccounts' 
            });
            
            this.wallet.connected = true;
            this.wallet.address = accounts[0];
            this.wallet.provider = window.ethereum;
            
            // Load USDC balance
            await this.loadUSDCBalance();
            
            // Update UI
            this.updateWalletUI();
            
            // Enable generate button
            document.getElementById('generateBlendBtn').disabled = false;
            
            console.log('✅ Wallet connected:', this.wallet.address);
            this.showSuccess('Wallet connected successfully');
            
        } catch (error) {
            console.error('❌ Failed to connect wallet:', error);
            this.showError('Failed to connect wallet: ' + error.message);
        }
    }
    
    /**
     * Load USDC balance from contract
     */
    async loadUSDCBalance() {
        try {
            // USDC contract ABI (balanceOf function only)
            const usdcABI = [
                {
                    "constant": true,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                }
            ];
            
            const web3 = new Web3(this.wallet.provider);
            const usdcContract = new web3.eth.Contract(usdcABI, this.config.USDC_ADDRESS);
            
            const balanceWei = await usdcContract.methods.balanceOf(this.wallet.address).call();
            
            // USDC has 6 decimals
            this.wallet.balance = parseFloat(balanceWei) / 1e6;
            
            this.updateWalletUI();
            
            console.log(`💰 USDC Balance: ${this.wallet.balance.toFixed(2)} USDC`);
            
        } catch (error) {
            console.warn('⚠️ Failed to load USDC balance:', error);
            this.wallet.balance = 0; // Fallback
        }
    }
    
    /**
     * Disconnect wallet
     */
    disconnectWallet() {
        this.wallet = {
            connected: false,
            address: null,
            balance: 0,
            provider: null
        };
        
        this.updateWalletUI();
        document.getElementById('generateBlendBtn').disabled = true;
        
        console.log('🔌 Wallet disconnected');
        this.showSuccess('Wallet disconnected');
    }
    
    /**
     * Update wallet UI
     */
    updateWalletUI() {
        const connectBtn = document.getElementById('connectWalletBtn');
        const connectedSection = document.getElementById('walletConnected');
        const addressEl = document.getElementById('walletAddress');
        const balanceEl = document.getElementById('walletBalance');
        
        if (this.wallet.connected) {
            connectBtn.style.display = 'none';
            connectedSection.style.display = 'flex';
            
            // Truncate address: 0x1234...5678
            const truncated = `${this.wallet.address.slice(0, 6)}...${this.wallet.address.slice(-4)}`;
            addressEl.textContent = truncated;
            balanceEl.textContent = `${this.wallet.balance.toFixed(2)} USDC`;
        } else {
            connectBtn.style.display = 'block';
            connectedSection.style.display = 'none';
        }
    }
    
    /**
     * Update cost estimate
     */
    updateCostEstimate() {
        const prompt = document.getElementById('blendPrompt').value;
        const quality = document.getElementById('blendQuality').value;
        
        if (!prompt) {
            document.getElementById('estimatedCost').textContent = '~0.00 USDC';
            return;
        }
        
        // Estimate based on prompt length and quality
        const pricing = {
            low: 0.01,
            medium: 0.05,
            high: 0.10,
            ultra: 0.25
        };
        
        const basePrice = pricing[quality] || 0.05;
        const estimatedDuration = 5; // Assume 5 seconds
        const complexity = Math.min(prompt.length / 50, 2); // Length-based complexity
        
        const cost = basePrice * estimatedDuration * complexity;
        
        document.getElementById('estimatedCost').textContent = `~${cost.toFixed(3)} USDC`;
    }
    
    /**
     * Generate blend from prompt
     */
    async generateBlendFromPrompt() {
        if (!this.wallet.connected) {
            this.showError('Please connect wallet first');
            return;
        }
        
        const prompt = document.getElementById('blendPrompt').value;
        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }
        
        const quality = document.getElementById('blendQuality').value;
        const statusEl = document.getElementById('paymentStatus');
        
        try {
            // Step 1: Analyze prompt
            statusEl.style.display = 'block';
            statusEl.className = 'payment-status processing';
            statusEl.querySelector('.status-text').textContent = '🔍 Analyzing prompt...';
            
            const analysisResponse = await fetch(`${this.config.API_BASE_URL}/api/prompts/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    prompt,
                    wallet: this.wallet.address 
                })
            });
            
            if (!analysisResponse.ok) {
                throw new Error(`Analysis failed: ${analysisResponse.status}`);
            }
            
            const analysis = await analysisResponse.json();
            
            // Step 2: Calculate cost
            const cost = this.calculateBlendCost(analysis, quality);
            
            statusEl.querySelector('.status-text').textContent = `💰 Creating payment (${cost.toFixed(3)} USDC)...`;
            
            // Step 3: Create x402 payment proof
            const paymentProof = await this.createX402Payment(cost, analysis.blend_id);
            
            statusEl.querySelector('.status-text').textContent = '🎨 Initiating blend generation...';
            
            // Step 4: Start blend generation (get streaming endpoint)
            const blendResponse = await fetch(`${this.config.API_BASE_URL}/api/motions/blend/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Payment': paymentProof
                },
                body: JSON.stringify({
                    prompt,
                    analysis,
                    quality,
                    wallet: this.wallet.address
                })
            });
            
            if (!blendResponse.ok) {
                throw new Error(`Blend generation failed: ${blendResponse.status}`);
            }
            
            const initResult = await blendResponse.json();
            console.log('Blend initiated:', initResult);
            
            // Step 5: Connect to streaming endpoint
            statusEl.querySelector('.status-text').textContent = '🔄 Streaming blend data...';
            
            const streamUrl = `${this.config.API_BASE_URL}${initResult.stream_url}`;
            const eventSource = new EventSource(streamUrl);
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Blend update:', data);
                
                // Update status
                const stepName = data.current_step_name || data.current_step;
                statusEl.querySelector('.status-text').textContent = `🔄 ${stepName} (${data.progress}/${data.total_steps})`;
                
                // Update metrics dashboard if metrics are available
                if (data.metrics && window.metricsDashboard) {
                    window.metricsDashboard.updateMetrics(data.metrics);
                }
                
                // Update workflow visualizer
                if (window.workflowVisualizer) {
                    window.workflowVisualizer.updateWorkflowState(data);
                }
                
                // Monitor video generation if available
                if (data.video && window.FEATURE_FLAGS?.ENABLE_VEO_GENERATION) {
                    const videoBtn = document.getElementById('viewBlendBtn');
                    const videoBadge = document.getElementById('viewBlendBadge');
                    
                    if (data.video.status === 'processing') {
                        if (videoBtn) {
                            videoBtn.style.display = 'flex';
                            if (videoBadge) videoBadge.textContent = `${data.video.progress || 0}%`;
                        }
                        
                        // Poll video status if operation_name provided
                        if (data.video.operation_name && !this._videoPolling) {
                            this._videoPolling = true;
                            this.pollVideoStatus(data.video.operation_name, (videoUpdate) => {
                                console.log('Video status update:', videoUpdate);
                                
                                if (videoUpdate.status === 'completed') {
                                    this._videoPolling = false;
                                    if (videoBadge) videoBadge.textContent = 'Ready';
                                    
                                    // Store video data
                                    if (!this.currentBlend) this.currentBlend = {};
                                    this.currentBlend.video = {
                                        ...videoUpdate,
                                        blend_name: `${analysis.motions[0]} → ${analysis.motions[1]}`,
                                        duration: '8s',
                                        resolution: '720p'
                                    };
                                    
                                    // Auto-show modal
                                    this.showVideoModal(this.currentBlend.video);
                                } else if (videoUpdate.status === 'failed') {
                                    this._videoPolling = false;
                                    if (videoBadge) videoBadge.textContent = 'Failed';
                                }
                            });
                        }
                    } else if (data.video.status === 'completed') {
                        if (videoBtn) {
                            videoBtn.style.display = 'flex';
                            if (videoBadge) videoBadge.textContent = 'Ready';
                        }
                        
                        // Store video data
                        if (!this.currentBlend) this.currentBlend = {};
                        this.currentBlend.video = {
                            ...data.video,
                            blend_name: `${analysis.motions[0]} → ${analysis.motions[1]}`,
                            duration: '8s',
                            resolution: '720p'
                        };
                    }
                }
                
                // Handle completion
                if (data.status === 'completed') {
                    eventSource.close();
                    
                    // Apply blend to scene
                    if (data.blend) {
                        this.applyGeneratedBlend(data.blend);
                    }
                    
                    // Record transaction
                    if (data.settlement) {
                        this.recordTransaction({
                            type: 'blend_generation',
                            prompt,
                            cost: cost.toFixed(3),
                            tx_hash: data.settlement.tx_hash,
                            timestamp: Date.now()
                        });
                    }
                    
                    // Refresh balance
                    this.loadUSDCBalance();
                    
                    // Success!
                    statusEl.className = 'payment-status success';
                    statusEl.querySelector('.status-text').textContent = '✅ Blend generated successfully!';
                    
                    setTimeout(() => {
                        statusEl.style.display = 'none';
                    }, 3000);
                    
                    this.showSuccess(`Blend generated: ${data.blend_id}`);
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('EventSource error:', error);
                eventSource.close();
                statusEl.className = 'payment-status error';
                statusEl.querySelector('.status-text').textContent = '❌ Stream connection lost';
            };
            
        } catch (error) {
            console.error('❌ Failed to generate blend:', error);
            statusEl.className = 'payment-status error';
            statusEl.querySelector('.status-text').textContent = `❌ ${error.message}`;
            
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 5000);
        }
    }
    
    /**
     * Calculate blend cost
     */
    calculateBlendCost(analysis, quality) {
        const pricing = {
            low: 0.01,
            medium: 0.05,
            high: 0.10,
            ultra: 0.25
        };
        
        const basePrice = pricing[quality] || 0.05;
        const duration = analysis.estimated_duration || 5;
        const motionCount = analysis.motions?.length || 2;
        const complexity = analysis.complexity || 1.0;
        
        return basePrice * duration * Math.sqrt(motionCount) * complexity;
    }
    
    /**
     * Create x402 payment proof
     */
    async createX402Payment(amount, resourceId) {
        const paymentData = {
            payTo: this.config.PLATFORM_ADDRESS,
            amount: amount.toString(),
            token: this.config.USDC_ADDRESS,
            chain: 'arc-testnet',
            resourceId: resourceId,
            userAddress: this.wallet.address,
            timestamp: Date.now()
        };
        
        // Sign payment data with wallet
        const message = JSON.stringify(paymentData);
        const signature = await window.ethereum.request({
            method: 'personal_sign',
            params: [message, this.wallet.address]
        });
        
        // Create payment proof (base64 encoded)
        const proof = btoa(JSON.stringify({
            ...paymentData,
            signature
        }));
        
        return proof;
    }
    
    /**
     * Apply generated blend to scene
     */
    applyGeneratedBlend(blend) {
        console.log('[MotionVisualizer] Entry: applyGeneratedBlend', blend);
        
        // Extract motion info
        const motions = blend.motions || [];
        const weights = blend.blend_weights || [];
        const blendFrames = blend.blend_frames || { start: 0, end: 0 };
        
        if (motions.length >= 2) {
            // Find motion objects
            const motionA = this.motionLibrary.find(m => 
                m.name.toLowerCase().includes(motions[0].toLowerCase())
            ) || this.motionLibrary[0];
            
            const motionB = this.motionLibrary.find(m => 
                m.name.toLowerCase().includes(motions[1].toLowerCase())
            ) || this.motionLibrary[1];
            
            const weight = weights[1] || 0.5;
            
            // Apply blend
            this.visualizeBlend(motionA, motionB, weight);
            
            // Update timeline with blend frame overlay
            this.updateTimelineWithBlendFrames([
                { 
                    name: motionA.name, 
                    weight: 1 - weight, 
                    color: '#6366f1',
                    duration: blend.duration || 5.0
                },
                { 
                    name: motionB.name, 
                    weight: weight, 
                    color: '#8b5cf6',
                    duration: blend.duration || 5.0
                }
            ], blendFrames);
            
            // Load metrics if available
            if (blend.metrics) {
                this.loadMetricsToDashboard(blend.metrics);
            }
        }
        
        this.stats.blendCount++;
        this.updateStats();
        
        console.log('[MotionVisualizer] Exit: applyGeneratedBlend - Blend applied to timeline');
    }
    
    /**
     * Record transaction
     */
    recordTransaction(transaction) {
        this.transactions.unshift(transaction);
        
        // Keep only last 10 transactions
        if (this.transactions.length > 10) {
            this.transactions = this.transactions.slice(0, 10);
        }
        
        this.displayTransactionHistory();
    }
    
    /**
     * Display transaction history
     */
    displayTransactionHistory() {
        const container = document.getElementById('transactionHistory');
        
        if (this.transactions.length === 0) {
            container.innerHTML = '<div class="empty-state">No transactions yet</div>';
            return;
        }
        
        container.innerHTML = this.transactions.map(tx => `
            <div class="transaction-item">
                <div>
                    <div class="tx-type">${tx.type.replace('_', ' ')}</div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary);">
                        ${new Date(tx.timestamp).toLocaleTimeString()}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="tx-cost">${tx.cost} USDC</div>
                    <a href="#" class="tx-link" title="${tx.tx_hash}">
                        ${tx.tx_hash.slice(0, 8)}...
                    </a>
                </div>
            </div>
        `).join('');
    }
    
    /**
     * Spawn NPC
     */
    async spawnNPC() {
        const characterType = document.getElementById('characterType').value;
        const energyLevel = parseInt(document.getElementById('energyLevel').value);
        
        try {
            // Use current blend if available
            const motionId = this.currentBlend?.motionA?.id || 'default';
            
            const response = await fetch(`${this.config.API_BASE_URL}/api/npcs/spawn`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    motion_id: motionId,
                    character_type: characterType,
                    energy_level: energyLevel
                })
            });
            
            if (!response.ok) {
                throw new Error(`Spawn failed: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('NPC spawn initiated:', result);
            
            // Connect to streaming endpoint
            const streamUrl = `${this.config.API_BASE_URL}${result.stream_url}`;
            const eventSource = new EventSource(streamUrl);
            
            this.showInfo('Spawning NPC...');
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('NPC spawn update:', data);
                
                // Update workflow visualizer
                if (window.workflowVisualizer) {
                    window.workflowVisualizer.updateWorkflowState(data);
                }
                
                // Handle completion
                if (data.status === 'completed') {
                    eventSource.close();
                    
                    // Render NPC in scene
                    this.renderNPC(data.npc_id, characterType, energyLevel);
                    
                    this.stats.npcCount++;
                    this.updateStats();
                    
                    this.showSuccess(`NPC spawned: ${data.npc_id}`);
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('NPC spawn stream error:', error);
                eventSource.close();
                this.showError('NPC spawn stream connection lost');
            };
            
        } catch (error) {
            console.error('❌ Failed to spawn NPC:', error);
            
            // Fallback: spawn NPC locally without backend
            const npcId = `npc-${Date.now()}`;
            this.renderNPC(npcId, characterType, energyLevel);
            this.stats.npcCount++;
            this.updateStats();
            this.showSuccess(`NPC spawned locally: ${npcId}`);
        }
    }
    
    /**
     * Render NPC in 3D scene
     */
    renderNPC(npcId, characterType, energyLevel) {
        // Create NPC mesh (smaller version of character)
        const scale = 0.6;
        const color = new THREE.Color().setHSL(Math.random(), 0.7, 0.5);
        
        const group = new THREE.Group();
        group.scale.set(scale, scale, scale);
        
        // Body
        const bodyGeometry = new THREE.BoxGeometry(0.8, 1.2, 0.4);
        const bodyMaterial = new THREE.MeshPhongMaterial({ 
            color,
            emissive: new THREE.Color(color).multiplyScalar(0.3)
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 1.5;
        body.castShadow = true;
        group.add(body);
        
        // Head
        const headGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const head = new THREE.Mesh(headGeometry, bodyMaterial);
        head.position.y = 2.4;
        head.castShadow = true;
        group.add(head);
        
        // Position randomly in scene
        const angle = Math.random() * Math.PI * 2;
        const radius = 3 + Math.random() * 2;
        group.position.set(
            Math.cos(angle) * radius,
            0,
            Math.sin(angle) * radius
        );
        
        // Add to scene
        this.scene.add(group);
        
        // Store NPC data
        this.npcs.push({
            id: npcId,
            mesh: group,
            characterType,
            energyLevel,
            createdAt: Date.now()
        });
        
        // Add wandering animation
        this.addNPCWandering(group);
        
        console.log(`✅ NPC rendered: ${npcId}`);
    }
    
    /**
     * Add wandering animation to NPC
     */
    addNPCWandering(npcMesh) {
        const wander = () => {
            const time = Date.now() * 0.0005;
            const radius = 3;
            npcMesh.position.x = Math.cos(time + npcMesh.uuid) * radius;
            npcMesh.position.z = Math.sin(time + npcMesh.uuid) * radius;
            npcMesh.rotation.y = time + npcMesh.uuid;
        };
        
        // Update in animation loop
        npcMesh.userData.update = wander;
    }
    
    /**
     * Toggle auto-spawn
     */
    toggleAutoSpawn(enabled) {
        if (enabled) {
            const interval = parseInt(document.getElementById('spawnInterval').value) * 1000;
            
            this.autoSpawnInterval = setInterval(() => {
                this.spawnNPC();
            }, interval);
            
            console.log(`✅ Auto-spawn enabled (every ${interval/1000}s)`);
        } else {
            if (this.autoSpawnInterval) {
                clearInterval(this.autoSpawnInterval);
                this.autoSpawnInterval = null;
            }
            console.log('⏸️ Auto-spawn disabled');
        }
    }
    
    /**
     * Playback controls
     */
    play() {
        this.isPlaying = true;
        console.log('▶ Playing');
    }
    
    pause() {
        this.isPlaying = false;
        console.log('⏸ Paused');
    }
    
    stop() {
        this.isPlaying = false;
        this.currentTime = 0;
        document.getElementById('playhead').style.left = '0%';
        console.log('⏹ Stopped');
    }
    
    /**
     * Reset camera
     */
    resetCamera() {
        this.camera.position.set(5, 3, 5);
        this.camera.lookAt(0, 1, 0);
        this.controls.target.set(0, 1, 0);
        this.controls.update();
    }
    
    /**
     * Toggle grid
     */
    toggleGrid() {
        this.gridHelper.visible = !this.gridHelper.visible;
    }
    
    /**
     * Export blend data
     */
    exportBlend() {
        if (!this.currentBlend) {
            this.showError('No blend to export');
            return;
        }
        
        const data = {
            blend_id: `blend-${Date.now()}`,
            motions: [
                { name: this.currentBlend.motionA.name, weight: 1 - this.currentBlend.weight },
                { name: this.currentBlend.motionB.name, weight: this.currentBlend.weight }
            ],
            timestamp: this.currentBlend.timestamp,
            metadata: {
                created_by: this.wallet.address || 'anonymous',
                tool: 'Kinetic Ledger Visualizer v2.0'
            }
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `blend-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showSuccess('Blend data exported');
    }
    
    /**
     * Mint NFT
     */
    async mintNFT() {
        if (!this.wallet.connected) {
            this.showError('Please connect wallet first');
            return;
        }
        
        if (!this.currentBlend) {
            this.showError('No blend to mint');
            return;
        }
        
        try {
            const response = await fetch(`${this.config.API_BASE_URL}/api/motions/mint`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    blend: this.currentBlend,
                    wallet: this.wallet.address
                })
            });
            
            if (!response.ok) {
                throw new Error(`Mint failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            this.showSuccess(`NFT minted: ${result.token_id}`);
            this.recordTransaction({
                type: 'nft_mint',
                cost: '0.00',
                tx_hash: result.tx_hash,
                timestamp: Date.now()
            });
            
        } catch (error) {
            console.error('❌ Failed to mint NFT:', error);
            this.showError('Minting failed: ' + error.message);
        }
    }
    
    /**
     * Filter motions
     */
    filterMotions(query) {
        const cards = document.querySelectorAll('.motion-card');
        const lowerQuery = query.toLowerCase();
        
        cards.forEach(card => {
            const name = card.querySelector('.motion-name').textContent.toLowerCase();
            const tags = Array.from(card.querySelectorAll('.tag'))
                .map(tag => tag.textContent.toLowerCase());
            
            const matches = name.includes(lowerQuery) || 
                          tags.some(tag => tag.includes(lowerQuery));
            
            card.style.display = matches ? 'block' : 'none';
        });
    }
    
    /**
     * Apply filter
     */
    applyFilter(filter) {
        // Update active button
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        // Filter cards
        const cards = document.querySelectorAll('.motion-card');
        
        cards.forEach(card => {
            if (filter === 'all') {
                card.style.display = 'block';
            } else {
                const tags = Array.from(card.querySelectorAll('.tag'))
                    .map(tag => tag.textContent.toLowerCase());
                
                card.style.display = tags.includes(filter) ? 'block' : 'none';
            }
        });
    }
    
    /**
     * Update stats display
     */
    updateStats() {
        document.getElementById('motionCount').textContent = this.stats.motionCount;
        document.getElementById('npcCount').textContent = this.stats.npcCount;
        document.getElementById('blendCount').textContent = this.stats.blendCount;
    }
    
    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Update NPCs
        this.npcs.forEach(npc => {
            if (npc.mesh.userData.update) {
                npc.mesh.userData.update();
            }
        });
        
        // Update playhead if playing
        if (this.isPlaying) {
            this.currentTime += 0.016; // ~60fps
            const progress = (this.currentTime % 10) / 10; // 10 second loop
            document.getElementById('playhead').style.left = `${progress * 100}%`;
            
            document.getElementById('currentTime').textContent = this.formatTime(this.currentTime);
        }
        
        // Update FPS
        this.updateFPS();
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    /**
     * Update FPS counter
     */
    updateFPS() {
        const now = performance.now();
        const delta = now - this.lastFrameTime;
        this.lastFrameTime = now;
        
        const fps = 1000 / delta;
        this.fpsHistory.push(fps);
        
        if (this.fpsHistory.length > 60) {
            this.fpsHistory.shift();
        }
        
        const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
        document.getElementById('fpsCounter').textContent = `${Math.round(avgFPS)} FPS`;
    }
    
    /**
     * Format time
     */
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * Handle window resize
     */
    onWindowResize() {
        const container = this.renderer.domElement.parentElement;
        
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
    
    /**
     * Show success message
     */
    showSuccess(message) {
        console.log('✅', message);
        // Could add toast notification here
    }
    
    /**
     * Show error message
     */
    showError(message) {
        console.error('❌', message);
        alert(message); // Simple alert for now
    }
    
    /**
     * Show info message
     */
    showInfo(message) {
        console.log('ℹ️', message);
        // Could add toast notification here
    }
    
    /**
     * Load FBX file and extract keyframe data
     */
    async loadFBXFile(fbxPath) {
        if (!window.FEATURE_FLAGS?.ENABLE_FBX_LOADER) {
            console.warn('[FBX] FBXLoader feature disabled');
            return null;
        }
        
        return new Promise((resolve, reject) => {
            const loader = new THREE.FBXLoader();
            
            loader.load(
                fbxPath,
                (fbx) => {
                    console.log(`[FBX] Loaded: ${fbxPath}`, fbx);
                    
                    // Extract animation data
                    const keyframes = [];
                    if (fbx.animations && fbx.animations.length > 0) {
                        const clip = fbx.animations[0];
                        const duration = clip.duration;
                        const fps = 30; // Mixamo standard
                        const frameCount = Math.floor(duration * fps);
                        
                        for (let i = 0; i < frameCount; i++) {
                            keyframes.push({
                                time: i / fps,
                                frame: i
                            });
                        }
                    }
                    
                    resolve({
                        fbx,
                        keyframes,
                        duration: fbx.animations[0]?.duration || 0,
                        fps: 30
                    });
                },
                (progress) => {
                    console.log(`[FBX] Loading progress: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                },
                (error) => {
                    console.error('[FBX] Load error:', error);
                    reject(error);
                }
            );
        });
    }
    
    /**
     * Show video modal with blend preview
     */
    showVideoModal(videoData) {
        if (!window.FEATURE_FLAGS?.ENABLE_VIDEO_MODAL) {
            console.warn('[VideoModal] Feature disabled');
            return;
        }
        
        const modal = document.getElementById('videoModal');
        const videoPlayer = document.getElementById('videoPlayer');
        const videoSource = document.getElementById('videoSource');
        const videoStatus = document.getElementById('videoStatus');
        const videoPlayerContainer = document.getElementById('videoPlayerContainer');
        const videoBlendName = document.getElementById('videoBlendName');
        const videoDuration = document.getElementById('videoDuration');
        const videoResolution = document.getElementById('videoResolution');
        
        if (!modal) return;
        
        // Show modal
        modal.style.display = 'flex';
        
        if (videoData.status === 'completed' && videoData.video_uri) {
            // Video ready - show player
            videoStatus.style.display = 'none';
            videoPlayerContainer.style.display = 'block';
            
            videoSource.src = videoData.video_uri;
            videoPlayer.load();
            
            // Update metadata
            if (videoBlendName) videoBlendName.textContent = videoData.blend_name || 'Motion Blend';
            if (videoDuration) videoDuration.textContent = videoData.duration || '8s';
            if (videoResolution) videoResolution.textContent = videoData.resolution || '720p';
        } else {
            // Show loading state
            videoStatus.style.display = 'block';
            videoPlayerContainer.style.display = 'none';
            
            const statusText = document.getElementById('videoStatusText');
            const progressFill = document.getElementById('videoProgress');
            const progressText = document.getElementById('videoProgressText');
            
            if (statusText) statusText.textContent = videoData.status_message || 'Generating video...';
            if (progressFill) progressFill.style.width = `${videoData.progress || 0}%`;
            if (progressText) progressText.textContent = `${videoData.progress || 0}%`;
        }
    }
    
    /**
     * Close video modal
     */
    closeVideoModal() {
        const modal = document.getElementById('videoModal');
        const videoPlayer = document.getElementById('videoPlayer');
        
        if (modal) modal.style.display = 'none';
        if (videoPlayer) videoPlayer.pause();
    }
    
    /**
     * Poll video generation status
     */
    async pollVideoStatus(operationName, onUpdate) {
        if (!window.FEATURE_FLAGS?.ENABLE_VEO_GENERATION) {
            console.warn('[Veo] Video generation feature disabled');
            return;
        }
        
        const maxAttempts = 60; // 5 minutes max (5s intervals)
        let attempts = 0;
        
        const poll = async () => {
            try {
                const response = await fetch(`${this.config.API_BASE_URL}/api/video/status/${encodeURIComponent(operationName)}`);
                const data = await response.json();
                
                if (onUpdate) onUpdate(data);
                
                if (data.status === 'completed') {
                    return data;
                } else if (data.status === 'failed') {
                    throw new Error(data.error || 'Video generation failed');
                } else if (attempts++ < maxAttempts) {
                    setTimeout(poll, 5000); // Poll every 5 seconds
                } else {
                    throw new Error('Video generation timeout');
                }
            } catch (error) {
                console.error('[Veo] Polling error:', error);
                if (onUpdate) onUpdate({ status: 'failed', error: error.message });
            }
        };
        
        poll();
    }
    
    /**
     * Apply feature flags to UI
     */
    applyFeatureFlags() {
        const flags = window.FEATURE_FLAGS || {};
        
        // Timeline-only mode
        if (flags.SHOW_TIMELINE_ONLY) {
            document.body.classList.add('timeline-only-mode');
            console.log('[FeatureFlags] Timeline-only mode enabled');
        }
        
        // Hide wallet section
        if (!flags.SHOW_WALLET) {
            const walletSection = document.querySelector('.wallet-section');
            if (walletSection) walletSection.style.display = 'none';
        }
        
        // Hide metrics dashboard
        if (!flags.SHOW_METRICS_DASHBOARD) {
            const dashboard = document.querySelector('.metrics-dashboard-compact');
            if (dashboard) dashboard.style.display = 'none';
        }
        
        console.log('[FeatureFlags] Applied:', flags);
    }
}

// Make MotionVisualizer globally accessible
window.MotionVisualizer = MotionVisualizer;
