# UI Visualizer Implementation Progress

**Created:** January 9, 2026  
**Status:** ✅ Foundation Complete, ⚠️ JavaScript Implementation Pending

---

## ✅ Completed Tasks

### 1. Mixamo Dataset Integration
- [x] Cloned `mixamo_anims_downloader` repository
- [x] Created Python download script: [scripts/download_mixamo.py](../scripts/download_mixamo.py)
- [x] Generated browser-ready download script: `data/mixamo_anims/downloadAll_prepared.js`
- [x] Created output directory structure: `data/mixamo_anims/fbx/`
- [x] Implemented verification and Gemini upload functions

**Features:**
- Automated character ID injection
- Manifest generation for Gemini File API
- Batch upload to Gemini with 10-file test limit
- FBX file verification and metadata extraction

**Next Steps:**
1. Visit [mixamo.com](https://mixamo.com) and log in
2. Open browser console (F12)
3. Copy contents of `data/mixamo_anims/downloadAll_prepared.js`
4. Paste into console to start batch download
5. Run `python scripts/download_mixamo.py --verify` to check downloads
6. Run `python scripts/download_mixamo.py --upload` to upload to Gemini

---

### 2. UI Foundation - HTML Structure
**File:** [src/kinetic_ledger/ui/index.html](../src/kinetic_ledger/ui/index.html)

- [x] Mixamo-inspired 3-column layout (library | viewport | controls)
- [x] Header with stats and wallet integration
- [x] Motion library sidebar with search and filters
- [x] Three.js canvas viewport with overlay controls
- [x] Control panel with prompt-based generation
- [x] Timeline footer with playback controls

**Key Components:**
```html
<!-- Header -->
- Logo + tagline
- Stats bar (motions, NPCs, blends)
- Wallet connect/disconnect

<!-- Main Content (3-column grid) -->
- Motion Library: search, filters, motion cards
- 3D Viewport: Three.js canvas, FPS counter, camera controls
- Control Panel: prompt textarea, quality selector, cost estimate, payment status

<!-- Timeline -->
- Playback controls (play/pause/stop)
- Timeline canvas with playhead
- Blend segments visualization
```

---

### 3. UI Styling - CSS
**File:** [src/kinetic_ledger/ui/styles.css](../src/kinetic_ledger/ui/styles.css)

- [x] Dark theme with purple/blue accents
- [x] CSS Grid layout (responsive design)
- [x] Smooth transitions and hover effects
- [x] Custom scrollbar styling
- [x] Wallet connection UI states
- [x] Payment status indicators (processing/success/error)
- [x] Motion card hover effects with glow
- [x] Timeline segment visualization

**Color Scheme:**
```css
--primary-color: #6366f1 (indigo)
--secondary-color: #8b5cf6 (purple)
--success-color: #10b981 (green)
--warning-color: #f59e0b (amber)
--error-color: #ef4444 (red)
--bg-dark: #0f172a (slate-900)
```

---

## ⚠️ Pending Tasks

### 4. JavaScript Implementation
**File:** `src/kinetic_ledger/ui/visualizer.js` (NOT YET CREATED)

Required implementation:

#### Core Classes
```javascript
class MotionVisualizer {
    constructor() {
        // Three.js scene setup
        // Wallet state
        // Motion library data
        // NPC tracking
    }
    
    init() {
        // Initialize Three.js scene
        // Load motion library
        // Setup event listeners
        // Start animation loop
    }
}
```

#### Wallet Integration Methods
- [ ] `connectWallet()` - MetaMask connection via `eth_requestAccounts`
- [ ] `loadUSDCBalance()` - Read USDC balance from contract (6 decimals)
- [ ] `disconnectWallet()` - Reset wallet state
- [ ] `updateWalletUI()` - Update address/balance display

#### Prompt-Based Generation
- [ ] `generateBlendFromPrompt()` - Full payment flow:
  1. Parse prompt via `/api/prompts/analyze`
  2. Calculate cost based on quality/duration/complexity
  3. Create x402 payment with wallet signature
  4. Submit to `/api/motions/blend/generate` with `X-Payment` header
  5. Apply blend to 3D scene
  6. Record transaction history

- [ ] `createX402Payment()` - Generate payment proof:
  ```javascript
  const paymentData = {
      payTo: '0x...platform',
      amount: '0.05',
      token: '0x...USDC',
      chain: 'arc-testnet',
      resourceId: blend_id,
      userAddress: wallet.address
  };
  
  const signature = await window.ethereum.request({
      method: 'personal_sign',
      params: [JSON.stringify(paymentData), wallet.address]
  });
  
  return btoa(JSON.stringify({...paymentData, signature}));
  ```

- [ ] `calculateBlendCost()` - Dynamic pricing:
  ```javascript
  const pricing = {low: 0.01, medium: 0.05, high: 0.10, ultra: 0.25};
  cost = basePrice * duration * sqrt(motionCount) * complexity;
  ```

#### Three.js Scene Management
- [ ] `setupScene()` - Camera, renderer, lighting, grid
- [ ] `loadMotionLibrary()` - Fetch from `/api/motions/library`
- [ ] `renderMotionCards()` - Populate library sidebar
- [ ] `applyBlend()` - Apply motion blend to character mesh
- [ ] `animate()` - Animation loop with FPS tracking

#### NPC Spawning
- [ ] `spawnNPC()` - POST to `/api/npcs/spawn`
- [ ] `renderNPC()` - Create 3D mesh in scene
- [ ] `toggleAutoSpawn()` - Perpetual spawning with interval
- [ ] `updateNPCState()` - Update energy/position

#### Manual Blend Controls
- [ ] `selectMotion()` - Click handler for motion cards
- [ ] `updateBlendWeight()` - Slider handler
- [ ] `applyManualBlend()` - Combine Motion A + Motion B

#### Transaction History
- [ ] `recordTransaction()` - Add to history
- [ ] `displayTransactionHistory()` - Render list with on-chain links

#### Timeline
- [ ] `updateTimeline()` - Render blend segments
- [ ] `updatePlayhead()` - Move playhead based on current time
- [ ] `playAnimation()` / `pauseAnimation()` / `stopAnimation()`

---

## 🔌 Backend API Integration Points

### Required Endpoints
The JavaScript implementation requires these backend endpoints:

#### 1. Motion Library
```python
@app.get("/api/motions/library")
async def get_motion_library():
    """Return catalog of available motions with metadata."""
    return {
        "motions": [
            {
                "id": "motion-1",
                "name": "Walking",
                "tags": ["locomotion", "neutral"],
                "duration": 5.0,
                "novelty": 0.3,
                "filepath": "data/mixamo_anims/fbx/Walking.fbx"
            },
            # ... more motions
        ]
    }
```

#### 2. Prompt Analysis
```python
@app.post("/api/prompts/analyze")
async def analyze_prompt(request: dict):
    """Parse natural language prompt into motion parameters."""
    prompt = request["prompt"]
    wallet = request["wallet"]
    
    # Use Gemini to parse prompt
    analysis = await gemini_analyzer.parse_motion_prompt(prompt)
    
    return {
        "blend_id": f"blend-{uuid.uuid4().hex[:8]}",
        "motions": ["walking", "dancing"],
        "weights": [0.3, 0.7],
        "estimated_duration": 8.0,
        "complexity": 1.5,
        "transition_type": "smooth"
    }
```

#### 3. Blend Generation
```python
@app.post("/api/motions/blend/generate")
async def generate_blend(
    request: dict,
    x_payment: str = Header(None, alias="X-Payment")
):
    """Generate motion blend with payment verification."""
    
    # Verify x402 payment
    payment_verified = await verify_x402_payment(x_payment)
    if not payment_verified:
        raise HTTPException(402, "Payment Required")
    
    # Generate blend using trustless agent
    result = await trustless_agent.execute_blend_workflow(
        prompt=request["prompt"],
        quality=request["quality"]
    )
    
    return {
        "status": "success",
        "blend": {
            "id": result.blend_id,
            "motions": result.motions,
            "blend_weights": result.blend_weights
        },
        "settlement": {
            "tx_hash": result.settlement_tx_hash,
            "amount_usdc": result.cost_usdc
        }
    }
```

#### 4. NPC Spawning
```python
@app.post("/api/npcs/spawn")
async def spawn_npc(request: dict):
    """Spawn NPC with motion blend."""
    
    npc_id = await arc_network.spawn_npc(
        motion_id=request["motion_id"],
        character_type=request["character_type"],
        energy_level=request["energy_level"]
    )
    
    return {
        "npc_id": npc_id,
        "status": "spawned",
        "motion_id": request["motion_id"]
    }
```

---

## 📋 Implementation Checklist

### Phase 1: Basic Scene Setup (Current)
- [x] HTML structure
- [x] CSS styling
- [ ] Create `visualizer.js`
- [ ] Three.js scene initialization
- [ ] Camera controls (OrbitControls)
- [ ] Grid helper and lighting
- [ ] FPS counter

### Phase 2: Motion Library
- [ ] Fetch motion library from backend
- [ ] Render motion cards in sidebar
- [ ] Search and filter functionality
- [ ] Motion selection (click to select)
- [ ] Display motion metadata (tags, duration, novelty)

### Phase 3: Wallet Integration
- [ ] MetaMask connection
- [ ] USDC balance reading
- [ ] Wallet UI updates (address truncation, balance formatting)
- [ ] Disconnect wallet

### Phase 4: Prompt-Based Generation
- [ ] Textarea input with suggestions
- [ ] Cost estimation based on quality
- [ ] x402 payment proof generation
- [ ] POST to `/api/prompts/analyze`
- [ ] POST to `/api/motions/blend/generate` with payment
- [ ] Payment status feedback (processing/success/error)
- [ ] Transaction history recording

### Phase 5: NPC Spawning
- [ ] Manual spawn with character type selector
- [ ] Auto-spawn with interval configuration
- [ ] Auto-payment option for spawns
- [ ] Render NPCs in 3D scene
- [ ] Update NPC energy levels

### Phase 6: Manual Blend Controls
- [ ] Dual motion selection (A + B)
- [ ] Blend weight slider
- [ ] Transition speed slider
- [ ] Apply blend button
- [ ] Visual feedback in timeline

### Phase 7: Timeline & Playback
- [ ] Render blend segments
- [ ] Playhead animation
- [ ] Play/pause/stop controls
- [ ] Current time display
- [ ] Click to scrub timeline

### Phase 8: Export & Minting
- [ ] Export blend data as JSON
- [ ] Mint Motion NFT on Arc Network
- [ ] Download FBX file

---

## 🚀 Quick Start Guide

### 1. Download Mixamo Animations
```bash
# Run the downloader script to get instructions
python scripts/download_mixamo.py

# Follow browser instructions to download animations
# Then verify:
python scripts/download_mixamo.py --verify

# Upload to Gemini:
export GEMINI_API_KEY="your-key-here"
python scripts/download_mixamo.py --upload
```

### 2. Start Backend Server
```bash
cd /workspaces/reimagined-umbrella

# Create backend API endpoints (if not done)
# Add routes to src/kinetic_ledger/api/server.py

# Start server
uvicorn src.kinetic_ledger.api.server:app --reload --port 8000
```

### 3. Open UI in Browser
```bash
# Serve UI files
# Option 1: Python HTTP server
cd src/kinetic_ledger/ui
python -m http.server 3000

# Option 2: Add route to FastAPI server
# In src/kinetic_ledger/api/server.py:
# from fastapi.responses import HTMLResponse, FileResponse
# 
# @app.get("/", response_class=HTMLResponse)
# async def serve_ui():
#     with open("src/kinetic_ledger/ui/index.html") as f:
#         return HTMLResponse(content=f.read())

# Then visit: http://localhost:8000
```

### 4. Connect Wallet & Test
1. Open browser to UI URL
2. Click "Connect Wallet"
3. Approve MetaMask connection
4. Enter prompt: "blend walk and run smoothly"
5. Click "Generate Blend & Pay"
6. Sign transaction
7. Watch blend appear in viewport

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER BROWSER                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   MetaMask   │  │  Three.js    │  │  UI (HTML/   │             │
│  │   Wallet     │  │  WebGL       │  │  CSS/JS)     │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                       │
│         │                 │                 │                       │
│         └─────────────────┴─────────────────┘                       │
│                           │                                         │
│                           │ HTTPS + x402 Payment                    │
│                           ▼                                         │
├─────────────────────────────────────────────────────────────────────┤
│                      BACKEND API SERVER                             │
│                  (FastAPI + Python)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  GET  /api/motions/library         → Motion catalog                │
│  POST /api/prompts/analyze         → Gemini prompt parsing         │
│  POST /api/motions/blend/generate  → Blend + payment settlement    │
│  POST /api/npcs/spawn              → Arc Network NPC spawning      │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Trustless Agent  │  │ Gemini Analyzer  │  │ Commerce Orch.   │ │
│  │ (Routing Logic)  │  │ (Prompt Parser)  │  │ (x402 Verifier)  │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘ │
│           │                     │                     │            │
│           └─────────────────────┴─────────────────────┘            │
│                                 │                                  │
├─────────────────────────────────┼──────────────────────────────────┤
│                                 ▼                                  │
│         ┌────────────────────────────────────────┐                 │
│         │      TRANSACTION ROUTING DECISION       │                 │
│         │   (On-Chain vs Off-Chain)               │                 │
│         └────────┬───────────────────────┬────────┘                 │
│                  │                       │                          │
│          On-Chain│                       │Off-Chain                 │
│                  ▼                       ▼                          │
│         ┌─────────────────┐    ┌─────────────────┐                 │
│         │  Arc Network L2  │    │  Circle Wallets │                 │
│         │  (NFT Minting)   │    │  (USDC Payments)│                 │
│         │  USDC Gas Token  │    │  Off-Chain Fast │                 │
│         └─────────────────┘    └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 File References

- **Mixamo Downloader:** [scripts/download_mixamo.py](../scripts/download_mixamo.py)
- **UI HTML:** [src/kinetic_ledger/ui/index.html](../src/kinetic_ledger/ui/index.html)
- **UI CSS:** [src/kinetic_ledger/ui/styles.css](../src/kinetic_ledger/ui/styles.css)
- **UI JavaScript:** `src/kinetic_ledger/ui/visualizer.js` (TODO)
- **Backend API:** [src/kinetic_ledger/api/server.py](../src/kinetic_ledger/api/server.py)
- **Trustless Agent:** [src/kinetic_ledger/services/trustless_agent.py](../src/kinetic_ledger/services/trustless_agent.py)
- **Commerce Orchestrator:** [src/kinetic_ledger/services/commerce_orchestrator.py](../src/kinetic_ledger/services/commerce_orchestrator.py)

---

**Last Updated:** January 9, 2026  
**Status:** UI foundation complete, JavaScript implementation next
