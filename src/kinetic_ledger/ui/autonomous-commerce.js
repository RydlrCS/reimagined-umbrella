/**
 * Kinetic Ledger - Autonomous Commerce Application
 * Live streaming blend strip marketplace on Arc Network
 * 
 * This module handles the Wallet & Commerce tab functionality:
 * - Circle Wallet creation/connection
 * - Unique blend strip generation with scoring
 * - Live commerce stream with polygon visualization
 * - Auto-generation mode for autonomous marketplace
 */

// ==================== STATE MANAGEMENT ====================
const CommerceState = {
    wallet: {
        connected: false,
        id: null,
        address: null,
        balance: 0,
        entitySecret: null
    },
    autoGen: {
        enabled: false,
        intervalId: null,
        generated: 0,
        sold: 0,
        earnings: 0,
        royalties: 0
    },
    currentBlend: null,
    transactions: [],
    metrics: {
        tps: 0,
        volume24h: 0,
        blendsMinted: 0,
        royaltiesPaid: 0
    },
    polygonNodes: [],
    canvasContext: null,
    animationFrame: null
};

// ==================== CONFIGURATION ====================
const CommerceConfig = {
    API_BASE: '/api/v2',
    ARC_CHAIN_ID: 1301,
    ARC_RPC: 'https://rpc.arcscan.io',
    USDC_DECIMALS: 6,
    BASE_PRICE: 1.0,
    UNIQUENESS_WEIGHTS: {
        frameRange: 0.30,
        motionCombo: 0.30,
        timing: 0.20,
        blendCurve: 0.20
    },
    MOTION_TYPES: [
        'walking', 'running', 'jumping', 'dancing', 'fighting',
        'capoeira', 'breakdance', 'hip-hop', 'robot', 'salsa',
        'kick', 'punch', 'idle', 'crouch', 'climb'
    ],
    MOTION_COLORS: {
        'walking': '#00D4AA',
        'running': '#FF6B35',
        'jumping': '#7B3FE4',
        'dancing': '#FFB800',
        'fighting': '#FF4757',
        'capoeira': '#FF6B35',
        'breakdance': '#7B3FE4',
        'hip-hop': '#00D4AA',
        'robot': '#2775CA',
        'salsa': '#FF4757',
        'kick': '#FFB800',
        'punch': '#FF6B35',
        'idle': '#8B8D97',
        'crouch': '#9B59B6',
        'climb': '#3498DB'
    },
    BLEND_CURVES: ['linear', 'ease-in', 'ease-out', 'ease-in-out', 'cubic', 'elastic']
};

// ==================== DOM ELEMENT REFERENCES ====================
// These are dynamically assigned after DOM loads
let Elements = {};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeApp();
    setupEventListeners();
    initializePolygonCanvas();
    startLiveUpdates();
});

/**
 * Initialize DOM element references for index-new.html structure
 */
function initializeElements() {
    Elements = {
        // Wallet Panel
        walletStatus: document.getElementById('walletStatus'),
        walletAddress: document.getElementById('walletAddress'),
        mainBalance: document.getElementById('mainBalance'),
        usdcBalance: document.getElementById('usdcBalance'),
        btnCreateWallet: document.getElementById('btnCreateWallet'),
        btnFaucet: document.getElementById('btnFaucet'),
        btnRefreshTx: document.getElementById('btnRefreshTx'),
        txList: document.getElementById('txList'),
        
        // Live Commerce Stream
        polygonCanvas: document.getElementById('polygonCanvas'),
        streamCanvas: document.getElementById('streamCanvas'),
        hexGrid: document.getElementById('hexGrid'),
        metricTPS: document.getElementById('metricTPS'),
        metricVolume: document.getElementById('metricVolume'),
        metricBlends: document.getElementById('metricBlends'),
        metricRoyalties: document.getElementById('metricRoyalties'),
        feedCount: document.getElementById('feedCount'),
        feedItems: document.getElementById('feedItems'),
        
        // Generator Panel
        autoGenerate: document.getElementById('autoGenerate'),
        autoStatus: document.getElementById('autoStatus'),
        frameStart: document.getElementById('frameStart'),
        frameEnd: document.getElementById('frameEnd'),
        blendComplexity: document.getElementById('blendComplexity'),
        priceMin: document.getElementById('priceMin'),
        priceMax: document.getElementById('priceMax'),
        genInterval: document.getElementById('genInterval'),
        btnGenerateOne: document.getElementById('btnGenerateOne'),
        
        // Generation Preview
        previewStatus: document.getElementById('previewStatus'),
        blendHash: document.getElementById('blendHash'),
        dataFrames: document.getElementById('dataFrames'),
        dataMotions: document.getElementById('dataMotions'),
        dataUniqueness: document.getElementById('dataUniqueness'),
        dataPrice: document.getElementById('dataPrice'),
        blendPreviewCanvas: document.getElementById('blendPreviewCanvas'),
        
        // Seller Stats
        statGenerated: document.getElementById('statGenerated'),
        statSold: document.getElementById('statSold'),
        statEarnings: document.getElementById('statEarnings'),
        statRoyalties: document.getElementById('statRoyalties'),
        
        // Marketplace Tab
        listingsGrid: document.getElementById('listingsGrid'),
        marketTotal: document.getElementById('marketTotal'),
        marketFloor: document.getElementById('marketFloor'),
        filterSort: document.getElementById('filterSort'),
        filterType: document.getElementById('filterType'),
        
        // Purchase Modal
        purchaseModal: document.getElementById('purchaseModal'),
        purchaseBlendId: document.getElementById('purchaseBlendId'),
        purchasePrice: document.getElementById('purchasePrice'),
        purchaseTotal: document.getElementById('purchaseTotal'),
        btnCancelPurchase: document.getElementById('btnCancelPurchase'),
        btnConfirmPurchase: document.getElementById('btnConfirmPurchase'),
        
        // Toast Container
        toastContainer: document.getElementById('toastContainer')
    };
}

function initializeApp() {
    console.log('🚀 Initializing Kinetic Ledger Autonomous Commerce...');
    
    // Check for existing wallet session
    const savedWallet = localStorage.getItem('kinetic_wallet');
    if (savedWallet) {
        try {
            const wallet = JSON.parse(savedWallet);
            AppState.wallet = { ...AppState.wallet, ...wallet, connected: true };
            updateWalletUI();
        } catch (e) {
            console.error('Failed to restore wallet session:', e);
        }
    }
    
    updateUniqueness();
}

function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    // Wallet connection
    Elements.connectWalletBtn.addEventListener('click', handleWalletConnection);
    
    // Segment management
    Elements.addSegmentBtn.addEventListener('click', addSegment);
    Elements.motionSelect.addEventListener('change', updateUniqueness);
    Elements.frameStart.addEventListener('input', updateUniqueness);
    Elements.frameEnd.addEventListener('input', updateUniqueness);
    Elements.blendCurve.addEventListener('change', updateUniqueness);
    Elements.transitionDuration.addEventListener('input', (e) => {
        Elements.transitionValue.textContent = e.target.value;
        updateUniqueness();
    });
    
    // Generate button
    Elements.generateBtn.addEventListener('click', generateAndMint);
}

// ==================== TAB MANAGEMENT ====================
function switchTab(tabId) {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.dataset.tab === tabId);
    });
}

// ==================== WALLET MANAGEMENT ====================
async function handleWalletConnection() {
    if (AppState.wallet.connected) {
        // Disconnect
        AppState.wallet = { connected: false, id: null, address: null, balance: 0 };
        localStorage.removeItem('kinetic_wallet');
        updateWalletUI();
        showToast('Wallet disconnected', 'info');
        return;
    }
    
    try {
        Elements.walletBtnText.textContent = 'Creating...';
        
        const response = await fetch('/api/v2/wallets/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                idempotency_key: `wallet-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
            })
        });
        
        if (!response.ok) throw new Error('Failed to create wallet');
        
        const data = await response.json();
        
        AppState.wallet = {
            connected: true,
            id: data.wallet_id,
            address: data.address,
            balance: 0
        };
        
        localStorage.setItem('kinetic_wallet', JSON.stringify(AppState.wallet));
        updateWalletUI();
        await refreshBalance();
        
        showToast('Wallet created on Arc Network!', 'success');
        
    } catch (error) {
        console.error('Wallet connection error:', error);
        showToast('Failed to create wallet: ' + error.message, 'error');
        Elements.walletBtnText.textContent = 'Connect Wallet';
    }
}

async function refreshBalance() {
    if (!AppState.wallet.id) return;
    
    try {
        const response = await fetch(`/api/v2/wallets/${AppState.wallet.id}/balance`);
        if (response.ok) {
            const data = await response.json();
            AppState.wallet.balance = parseFloat(data.balance) || 0;
            updateWalletUI();
        }
    } catch (error) {
        console.error('Failed to refresh balance:', error);
    }
}

function updateWalletUI() {
    if (AppState.wallet.connected) {
        Elements.walletBtnText.textContent = shortenAddress(AppState.wallet.address);
        Elements.totalBalance.textContent = AppState.wallet.balance.toFixed(2);
        Elements.walletAddress.textContent = AppState.wallet.address || '0x0000...0000';
        Elements.usdcBalance.textContent = `${AppState.wallet.balance.toFixed(2)} USDC`;
    } else {
        Elements.walletBtnText.textContent = 'Connect Wallet';
        Elements.totalBalance.textContent = '0.00';
        Elements.walletAddress.textContent = '0x0000...0000';
        Elements.usdcBalance.textContent = '0.00 USDC';
    }
}

function shortenAddress(address) {
    if (!address) return '0x0000...0000';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

// ==================== SEGMENT MANAGEMENT ====================
function addSegment() {
    const motion = Elements.motionSelect.value;
    const frameStart = parseInt(Elements.frameStart.value) || 0;
    const frameEnd = parseInt(Elements.frameEnd.value) || 30;
    
    if (!motion) {
        showToast('Please select a motion', 'error');
        return;
    }
    
    if (frameEnd <= frameStart) {
        showToast('End frame must be greater than start frame', 'error');
        return;
    }
    
    const segment = {
        id: `seg-${Date.now()}`,
        motion,
        frameStart,
        frameEnd,
        color: Config.MOTION_COLORS[motion] || '#7B3FE4'
    };
    
    AppState.segments.push(segment);
    renderTimeline();
    updateUniqueness();
    
    // Reset form
    Elements.motionSelect.value = '';
    Elements.frameStart.value = 0;
    Elements.frameEnd.value = 30;
    
    showToast(`Added ${motion} segment (${frameEnd - frameStart} frames)`, 'success');
}

function removeSegment(segmentId) {
    AppState.segments = AppState.segments.filter(s => s.id !== segmentId);
    renderTimeline();
    updateUniqueness();
}

function renderTimeline() {
    if (AppState.segments.length === 0) {
        Elements.timelineTrack.innerHTML = `
            <div style="color: var(--arc-text-muted); font-size: 13px; display: flex; align-items: center; justify-content: center; width: 100%;">
                Add motion segments above to build your blend strip
            </div>
        `;
        Elements.totalFrames.textContent = '0 frames | 0.0s';
        return;
    }
    
    let totalFrames = 0;
    const html = AppState.segments.map(seg => {
        const frames = seg.frameEnd - seg.frameStart;
        totalFrames += frames;
        return `
            <div class="segment-block" style="background: ${seg.color}; flex: ${frames};" draggable="true" data-id="${seg.id}">
                <div class="name">${seg.motion}</div>
                <div class="frames">${seg.frameStart}-${seg.frameEnd} (${frames}f)</div>
                <button class="remove" onclick="removeSegment('${seg.id}')">&times;</button>
            </div>
        `;
    }).join('');
    
    Elements.timelineTrack.innerHTML = html;
    Elements.totalFrames.textContent = `${totalFrames} frames | ${(totalFrames / 30).toFixed(1)}s`;
}

// ==================== UNIQUENESS CALCULATION ====================
function updateUniqueness() {
    // Frame Range Diversity (30% weight)
    // Higher diversity = more unique combinations of frame ranges
    let frameRangeScore = 0;
    if (AppState.segments.length > 0) {
        const ranges = AppState.segments.map(s => s.frameEnd - s.frameStart);
        const avgRange = ranges.reduce((a, b) => a + b, 0) / ranges.length;
        const variance = ranges.reduce((sum, r) => sum + Math.pow(r - avgRange, 2), 0) / ranges.length;
        frameRangeScore = Math.min(100, (variance / 100) * 100 + AppState.segments.length * 15);
    }
    
    // Motion Combination (30% weight)
    // More unique motions = higher score
    let motionComboScore = 0;
    if (AppState.segments.length > 0) {
        const uniqueMotions = new Set(AppState.segments.map(s => s.motion)).size;
        const totalMotions = Object.keys(Config.MOTION_COLORS).length;
        motionComboScore = (uniqueMotions / Math.min(AppState.segments.length, totalMotions)) * 100;
        // Bonus for rare combinations
        if (uniqueMotions >= 3) motionComboScore = Math.min(100, motionComboScore * 1.2);
    }
    
    // Timing Uniqueness (20% weight)
    // Based on transition duration and total length
    const transitionDuration = parseInt(Elements.transitionDuration.value);
    const timingScore = Math.min(100, (transitionDuration / 30) * 50 + AppState.segments.length * 20);
    
    // Blend Curve (20% weight)
    // Non-linear curves are more unique
    const curveScores = {
        'linear': 30,
        'smoothstep': 50,
        'ease-in': 60,
        'ease-out': 60,
        'bounce': 100
    };
    const blendCurveScore = curveScores[Elements.blendCurve.value] || 50;
    
    // Calculate total uniqueness
    const totalUniqueness = Math.round(
        frameRangeScore * 0.30 +
        motionComboScore * 0.30 +
        timingScore * 0.20 +
        blendCurveScore * 0.20
    );
    
    // Update state
    AppState.uniqueness = {
        total: totalUniqueness,
        frameRange: Math.round(frameRangeScore),
        motionCombo: Math.round(motionComboScore),
        timing: Math.round(timingScore),
        blendCurve: Math.round(blendCurveScore)
    };
    
    // Update UI
    Elements.uniquenessScore.textContent = `${totalUniqueness}%`;
    Elements.uniquenessFill.style.width = `${totalUniqueness}%`;
    Elements.frameRangeScore.textContent = `${Math.round(frameRangeScore)}%`;
    Elements.motionComboScore.textContent = `${Math.round(motionComboScore)}%`;
    Elements.timingScore.textContent = `${Math.round(timingScore)}%`;
    Elements.blendCurveScore.textContent = `${Math.round(blendCurveScore)}%`;
    
    // Update pricing
    const bonus = (totalUniqueness / 100) * Config.BASE_PRICE * Config.UNIQUENESS_MULTIPLIER;
    const total = Config.BASE_PRICE + bonus + 0.01;
    Elements.uniquenessBonus.textContent = `+${bonus.toFixed(2)} USDC`;
    Elements.totalPrice.textContent = `${total.toFixed(2)} USDC`;
    
    // Update uniqueness score color
    if (totalUniqueness >= 80) {
        Elements.uniquenessScore.style.color = '#00D4AA';
    } else if (totalUniqueness >= 50) {
        Elements.uniquenessScore.style.color = '#FFB800';
    } else {
        Elements.uniquenessScore.style.color = '#FF4757';
    }
}

// ==================== GENERATE & MINT ====================
async function generateAndMint() {
    if (!AppState.wallet.connected) {
        showToast('Please connect your wallet first', 'error');
        return;
    }
    
    if (AppState.segments.length === 0) {
        showToast('Please add at least one segment', 'error');
        return;
    }
    
    const btn = Elements.generateBtn;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    
    try {
        // Generate blend hash
        const blendData = {
            segments: AppState.segments,
            blendCurve: Elements.blendCurve.value,
            transitionDuration: parseInt(Elements.transitionDuration.value),
            uniqueness: AppState.uniqueness,
            creator: AppState.wallet.address,
            timestamp: Date.now()
        };
        
        const blendHash = await generateBlendHash(blendData);
        
        // Calculate price
        const bonus = (AppState.uniqueness.total / 100) * Config.BASE_PRICE * Config.UNIQUENESS_MULTIPLIER;
        const price = Config.BASE_PRICE + bonus;
        
        // Mint on Arc Network
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Minting on Arc...';
        
        const mintResponse = await fetch('/api/v2/motions/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                motion_id: blendHash,
                creator: AppState.wallet.address,
                metadata: blendData,
                price_usdc: price,
                uniqueness_score: AppState.uniqueness.total
            })
        });
        
        if (!mintResponse.ok) throw new Error('Failed to mint blend');
        
        const mintData = await mintResponse.json();
        
        // Add to marketplace
        const newBlend = {
            id: blendHash,
            hash: blendHash,
            segments: AppState.segments,
            uniqueness: AppState.uniqueness.total,
            price,
            creator: AppState.wallet.address,
            txHash: mintData.tx_hash || `0x${blendHash.slice(0, 64)}`,
            timestamp: Date.now()
        };
        
        AppState.marketplace.unshift(newBlend);
        renderMarketplace();
        
        // Add transaction to stream
        addTransaction({
            type: 'mint',
            hash: newBlend.txHash,
            amount: price,
            from: AppState.wallet.address,
            blendId: blendHash,
            timestamp: Date.now()
        });
        
        // Clear segments
        AppState.segments = [];
        renderTimeline();
        updateUniqueness();
        
        showToast(`Blend minted successfully! Hash: ${shortenAddress(blendHash)}`, 'success');
        
    } catch (error) {
        console.error('Mint error:', error);
        showToast('Failed to mint: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-magic"></i> Generate & Mint on Arc';
    }
}

async function generateBlendHash(data) {
    const encoder = new TextEncoder();
    const dataString = JSON.stringify(data);
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoder.encode(dataString));
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ==================== MARKETPLACE ====================
async function loadMarketplace() {
    try {
        const response = await fetch('/api/v2/marketplace/blends');
        if (response.ok) {
            const data = await response.json();
            AppState.marketplace = data.blends || [];
        }
    } catch (error) {
        console.log('Loading sample marketplace data...');
        // Generate sample blends for demo
        AppState.marketplace = generateSampleBlends();
    }
    
    renderMarketplace();
}

function generateSampleBlends() {
    const motions = ['capoeira', 'breakdance', 'hip-hop', 'robot', 'salsa', 'kick', 'punch'];
    const blends = [];
    
    for (let i = 0; i < 8; i++) {
        const numSegments = 2 + Math.floor(Math.random() * 3);
        const segments = [];
        
        for (let j = 0; j < numSegments; j++) {
            const motion = motions[Math.floor(Math.random() * motions.length)];
            segments.push({
                motion,
                frameStart: j * 30,
                frameEnd: (j + 1) * 30 + Math.floor(Math.random() * 15),
                color: Config.MOTION_COLORS[motion]
            });
        }
        
        const uniqueness = 40 + Math.floor(Math.random() * 50);
        const price = 5 + (uniqueness / 100) * 2.5;
        
        blends.push({
            id: `blend-${i}`,
            hash: Array.from(crypto.getRandomValues(new Uint8Array(32)))
                .map(b => b.toString(16).padStart(2, '0')).join(''),
            segments,
            uniqueness,
            price,
            creator: `0x${Math.random().toString(16).slice(2, 10)}...${Math.random().toString(16).slice(2, 6)}`,
            txHash: `0x${Math.random().toString(16).slice(2, 66)}`,
            timestamp: Date.now() - Math.floor(Math.random() * 86400000)
        });
    }
    
    return blends;
}

function renderMarketplace() {
    const html = AppState.marketplace.map(blend => `
        <div class="blend-card" data-id="${blend.id}">
            <div class="blend-preview">
                <div class="blend-strip-viz">
                    ${blend.segments.map(seg => `
                        <div class="frame-segment" style="background: ${seg.color}; flex: ${seg.frameEnd - seg.frameStart};"></div>
                    `).join('')}
                </div>
            </div>
            <div class="blend-info">
                <div class="blend-title">
                    Blend Strip
                    <span class="uniqueness-badge" style="background: ${blend.uniqueness >= 80 ? '#00D4AA' : blend.uniqueness >= 50 ? '#FFB800' : '#FF4757'}">
                        ${blend.uniqueness}% Unique
                    </span>
                </div>
                <div class="blend-meta">
                    <span><i class="fas fa-film"></i> ${blend.segments.length} segments</span>
                    <span><i class="fas fa-clock"></i> ${getTimeAgo(blend.timestamp)}</span>
                </div>
                <div class="blend-hash">
                    <i class="fas fa-fingerprint"></i> ${blend.hash.slice(0, 16)}...${blend.hash.slice(-8)}
                </div>
                <div class="blend-price">
                    <div class="price-tag">
                        <i class="fas fa-coins" style="color: #2775CA;"></i>
                        <span class="price-value">${blend.price.toFixed(2)}</span>
                        <span style="color: var(--arc-text-muted);">USDC</span>
                    </div>
                    <button class="buy-btn" onclick="buyBlend('${blend.id}')" ${!AppState.wallet.connected ? 'disabled' : ''}>
                        <i class="fas fa-shopping-cart"></i> Buy
                    </button>
                </div>
            </div>
        </div>
    `).join('');
    
    Elements.marketplaceGrid.innerHTML = html;
}

async function buyBlend(blendId) {
    const blend = AppState.marketplace.find(b => b.id === blendId);
    if (!blend) return;
    
    if (!AppState.wallet.connected) {
        showToast('Please connect your wallet first', 'error');
        return;
    }
    
    if (AppState.wallet.balance < blend.price) {
        showToast('Insufficient balance', 'error');
        return;
    }
    
    try {
        showToast('Processing purchase...', 'info');
        
        // x402 payment flow
        const response = await fetch('/api/v2/payments/settle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                buyer_wallet_id: AppState.wallet.id,
                motion_id: blend.id,
                amount_usdc: blend.price
            })
        });
        
        if (!response.ok) throw new Error('Payment failed');
        
        const data = await response.json();
        
        // Update balance
        AppState.wallet.balance -= blend.price;
        updateWalletUI();
        
        // Add transaction
        addTransaction({
            type: 'buy',
            hash: data.tx_hash || `0x${Date.now().toString(16)}`,
            amount: blend.price,
            from: AppState.wallet.address,
            blendId: blend.id,
            timestamp: Date.now()
        });
        
        showToast(`Successfully purchased blend for ${blend.price.toFixed(2)} USDC!`, 'success');
        
    } catch (error) {
        console.error('Purchase error:', error);
        showToast('Purchase failed: ' + error.message, 'error');
    }
}

// ==================== LIVE TRANSACTION STREAM ====================
function addTransaction(tx) {
    AppState.transactions.unshift(tx);
    if (AppState.transactions.length > 50) {
        AppState.transactions.pop();
    }
    
    renderTransactions();
    addPolygonNode(tx);
    updateMetrics();
}

function renderTransactions() {
    const html = AppState.transactions.slice(0, 10).map(tx => `
        <div class="tx-card ${tx.confirmed ? 'confirmed' : ''}">
            <div class="tx-header">
                <div class="tx-type">
                    <div class="tx-icon ${tx.type}">${getTxIcon(tx.type)}</div>
                    <div>
                        <div style="font-weight: 600;">${getTxLabel(tx.type)}</div>
                        <div style="font-size: 12px; color: var(--arc-text-muted);">${getTimeAgo(tx.timestamp)}</div>
                    </div>
                </div>
                <div class="tx-amount">${tx.type === 'sell' ? '+' : ''}${tx.amount.toFixed(2)} USDC</div>
            </div>
            <div class="tx-hash">
                <i class="fas fa-link"></i> ${tx.hash.slice(0, 20)}...${tx.hash.slice(-8)}
            </div>
            <div class="tx-meta">
                <span>${shortenAddress(tx.from)}</span>
                <span style="color: ${tx.confirmed ? 'var(--success)' : 'var(--warning)'}">
                    <i class="fas fa-${tx.confirmed ? 'check-circle' : 'clock'}"></i>
                    ${tx.confirmed ? 'Confirmed' : 'Pending'}
                </span>
            </div>
        </div>
    `).join('');
    
    Elements.txStream.innerHTML = html;
}

function getTxIcon(type) {
    const icons = {
        'buy': '<i class="fas fa-shopping-cart"></i>',
        'sell': '<i class="fas fa-tags"></i>',
        'mint': '<i class="fas fa-magic"></i>'
    };
    return icons[type] || '<i class="fas fa-exchange-alt"></i>';
}

function getTxLabel(type) {
    const labels = {
        'buy': 'Purchase',
        'sell': 'Sale',
        'mint': 'New Mint'
    };
    return labels[type] || 'Transaction';
}

// ==================== POLYGON VISUALIZATION ====================
let polygonCtx;
let animationFrame;

function initializePolygonCanvas() {
    const canvas = Elements.polygonCanvas;
    const container = canvas.parentElement;
    
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    polygonCtx = canvas.getContext('2d');
    
    // Generate initial nodes
    for (let i = 0; i < 15; i++) {
        addPolygonNode({
            type: ['buy', 'sell', 'mint'][Math.floor(Math.random() * 3)],
            amount: 5 + Math.random() * 20,
            timestamp: Date.now() - Math.random() * 10000
        });
    }
    
    animatePolygon();
}

function addPolygonNode(tx) {
    const canvas = Elements.polygonCanvas;
    
    const node = {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        radius: 5 + (tx.amount || 5) * 0.5,
        color: tx.type === 'buy' ? '#00D4AA' : tx.type === 'sell' ? '#7B3FE4' : '#FF6B35',
        alpha: 1,
        tx
    };
    
    AppState.polygonNodes.push(node);
    
    // Limit nodes
    if (AppState.polygonNodes.length > 30) {
        AppState.polygonNodes.shift();
    }
}

function animatePolygon() {
    const canvas = Elements.polygonCanvas;
    const ctx = polygonCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw connections
    ctx.strokeStyle = 'rgba(123, 63, 228, 0.1)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < AppState.polygonNodes.length; i++) {
        for (let j = i + 1; j < AppState.polygonNodes.length; j++) {
            const a = AppState.polygonNodes[i];
            const b = AppState.polygonNodes[j];
            const dist = Math.hypot(a.x - b.x, a.y - b.y);
            
            if (dist < 150) {
                ctx.globalAlpha = 1 - dist / 150;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }
        }
    }
    
    // Draw and update nodes
    AppState.polygonNodes.forEach(node => {
        // Update position
        node.x += node.vx;
        node.y += node.vy;
        
        // Bounce off walls
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
        
        // Draw node
        ctx.globalAlpha = node.alpha;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();
        
        // Draw glow
        const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, node.radius * 2
        );
        gradient.addColorStop(0, node.color + '40');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius * 2, 0, Math.PI * 2);
        ctx.fill();
    });
    
    ctx.globalAlpha = 1;
    animationFrame = requestAnimationFrame(animatePolygon);
}

// ==================== LIVE UPDATES ====================
function startLiveUpdates() {
    // Simulate live transactions
    setInterval(() => {
        if (Math.random() > 0.7) {
            simulateTransaction();
        }
    }, 3000);
    
    // Update metrics
    setInterval(updateMetrics, 5000);
    
    // Confirm pending transactions
    setInterval(() => {
        AppState.transactions.forEach(tx => {
            if (!tx.confirmed && Date.now() - tx.timestamp > 5000) {
                tx.confirmed = true;
            }
        });
        renderTransactions();
    }, 2000);
}

function simulateTransaction() {
    const types = ['buy', 'sell', 'mint'];
    const type = types[Math.floor(Math.random() * types.length)];
    
    const tx = {
        type,
        hash: `0x${Array.from(crypto.getRandomValues(new Uint8Array(32)))
            .map(b => b.toString(16).padStart(2, '0')).join('')}`,
        amount: 5 + Math.random() * 25,
        from: `0x${Math.random().toString(16).slice(2, 10)}...${Math.random().toString(16).slice(2, 6)}`,
        blendId: `blend-${Date.now()}`,
        timestamp: Date.now(),
        confirmed: false
    };
    
    addTransaction(tx);
}

function updateMetrics() {
    const confirmed = AppState.transactions.filter(tx => tx.confirmed);
    const volume24h = confirmed.reduce((sum, tx) => sum + tx.amount, 0);
    
    AppState.metrics = {
        activeBlends: AppState.marketplace.length,
        dailyVolume: volume24h,
        avgUniqueness: AppState.marketplace.length > 0 
            ? Math.round(AppState.marketplace.reduce((sum, b) => sum + b.uniqueness, 0) / AppState.marketplace.length)
            : 0,
        gasSponsored: confirmed.length * 0.01,
        lastBlock: Math.floor(Date.now() / 12000),
        pendingTx: AppState.transactions.filter(tx => !tx.confirmed).length
    };
    
    document.getElementById('activeBlends').textContent = AppState.metrics.activeBlends;
    document.getElementById('dailyVolume').textContent = `${AppState.metrics.dailyVolume.toFixed(2)} USDC`;
    document.getElementById('avgUniqueness').textContent = `${AppState.metrics.avgUniqueness}%`;
    document.getElementById('gasSponsored').textContent = `${AppState.metrics.gasSponsored.toFixed(3)} USDC`;
    document.getElementById('lastBlock').textContent = `#${AppState.metrics.lastBlock}`;
    document.getElementById('pendingTx').textContent = AppState.metrics.pendingTx;
}

// ==================== UTILITIES ====================
function getTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Wallet helper functions
function depositFunds() {
    showToast('Deposit feature coming soon! Use Arc faucet for testnet USDC.', 'info');
}

function withdrawFunds() {
    if (!AppState.wallet.connected) {
        showToast('Please connect wallet first', 'error');
        return;
    }
    showToast('Withdrawal feature coming soon!', 'info');
}

function viewOnExplorer() {
    if (AppState.wallet.address) {
        window.open(`https://explorer.arcscan.io/address/${AppState.wallet.address}`, '_blank');
    }
}

// Export for window scope
window.removeSegment = removeSegment;
window.buyBlend = buyBlend;
window.depositFunds = depositFunds;
window.withdrawFunds = withdrawFunds;
window.viewOnExplorer = viewOnExplorer;

console.log('✅ Kinetic Ledger Autonomous Commerce loaded');
