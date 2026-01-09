# Arc Network Integration Guide

## Overview

The Kinetic Ledger integrates with **Arc Network** (Circle's L1 blockchain) to manage on-chain NPC animation state and motion pack ownership for game engines. Arc uses **USDC as the native gas token**, making it ideal for gaming micropayments and creator royalties.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Game Engine (Unity/Unreal)                    │
│                                                                   │
│  ┌────────────┐  ┌───────────┐  ┌──────────────┐               │
│  │ NPC System │  │ Animation │  │ Blend System │               │
│  │            │──│  Manager  │──│              │               │
│  └────────────┘  └───────────┘  └──────────────┘               │
│         │              │                 │                       │
└─────────┼──────────────┼─────────────────┼───────────────────────┘
          │              │                 │
          ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Kinetic Ledger Services (Python)                    │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ Arc Network  │  │  Attestation │  │ Gemini Analyzer│        │
│  │   Service    │──│    Oracle    │──│                │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│         │                  │                    │                │
└─────────┼──────────────────┼────────────────────┼────────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Arc Network (Testnet)                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │        NPCMotionRegistry Smart Contract              │       │
│  │                                                       │       │
│  │  • Motion Pack Minting                               │       │
│  │  • NPC Spawning & State Management                   │       │
│  │  • Usage Tracking & Payments (USDC)                  │       │
│  │  • Creator Royalties (70% / 10% / 15% / 5%)          │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
│  RPC: https://rpc.testnet.arc.network                           │
│  Explorer: https://testnet.arcscan.app                          │
└─────────────────────────────────────────────────────────────────┘
```

## Smart Contract: NPCMotionRegistry

### Key Features

1. **Motion Pack Minting**
   - Stores canonical pack hash from attestation oracle
   - Links to IPFS for BVH/FBX data
   - Records style labels and NPC tags
   - Tracks creator for royalty distribution

2. **NPC State Management**
   - On-chain NPC spawning with initial motion
   - Real-time animation state updates
   - Energy/stamina tracking (0-100)
   - Blend state hash for synchronization

3. **Usage-Based Payments**
   - Per-second usage fees in USDC
   - Automatic royalty distribution:
     * 70% to motion creator
     * 10% to attestation oracle
     * 15% to platform
     * 5% to operations
   - Usage count tracking per motion

4. **Access Control**
   - Oracle authorization for minting
   - Game engine authorization for NPC operations
   - Owner-only administrative functions

## Deployment Guide

### Prerequisites

1. **Install Foundry**
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

2. **Get Testnet USDC**
   - Visit [https://faucet.circle.com](https://faucet.circle.com)
   - Select "Arc Testnet"
   - Request testnet USDC (used for gas)

3. **Configure Environment**
   ```bash
   cp .env.arc.example .env
   
   # Edit .env with your values:
   # - PRIVATE_KEY: Your wallet private key
   # - PLATFORM_ADDRESS: Platform wallet address
   # - OPS_ADDRESS: Operations wallet address
   # - MINT_PRICE: Cost to mint (default 1 USDC = 1000000)
   # - USAGE_FEE_PER_SECOND: Per-second fee (default 100 = 0.0001 USDC)
   ```

### Deploy Contract

```bash
# Make scripts executable
chmod +x scripts/deploy-arc.sh scripts/authorize-arc.sh

# Deploy NPCMotionRegistry
./scripts/deploy-arc.sh
```

**Expected Output:**
```
🚀 Deploying NPCMotionRegistry to Arc Testnet...

Configuration:
  RPC URL: https://rpc.testnet.arc.network
  Platform Address: 0x...
  Ops Address: 0x...
  Mint Price: 1000000 USDC
  Usage Fee: 100 USDC/second

Deployer: 0xB815A0c4bC23930119324d4359dB65e27A846A2d
Deployed to: 0x32368037b14819C9e5Dbe96b3d67C59b8c65c4BF
Transaction hash: 0xeba0fcb5e528d586db0aeb2465a8fad0299330a9773ca62818a1827560a67346

✅ Deployment complete!
```

### Authorize Oracle & Engines

```bash
# Set contract address in .env
export NPC_MOTION_REGISTRY_ADDRESS="0x32368037b14819C9e5Dbe96b3d67C59b8c65c4BF"
export ORACLE_ADDRESS="0x..."
export GAME_ENGINE_ADDRESSES="0x...,0x..."

# Authorize
./scripts/authorize-arc.sh
```

## Python Integration

### Installation

```bash
pip install web3 eth-account
```

### Basic Usage

```python
from kinetic_ledger.services.arc_network import ArcNetworkService
from kinetic_ledger.utils.canonicalize import keccak256

# Initialize service
arc = ArcNetworkService(
    rpc_url="https://rpc.testnet.arc.network",
    private_key=os.getenv("PRIVATE_KEY"),
    contract_address="0x32368037b14819C9e5Dbe96b3d67C59b8c65c4BF"
)

# Mint a motion pack (called by attestation oracle)
pack_hash = keccak256(canonical_pack_bytes)
result = arc.mint_motion_pack(
    pack_hash=pack_hash,
    creator_address="0x...",
    ipfs_uri="ipfs://QmX...",
    style_labels=["capoeira", "breakdance"],
    npc_tags=["agile", "evasive", "dynamic"]
)

print(f"Motion ID: {result['motion_id']}")
print(f"TX Hash: {result['tx_hash']}")

# Spawn an NPC (called by game engine)
npc_result = arc.spawn_npc(motion_id=result['motion_id'])
print(f"NPC ID: {npc_result['npc_id']}")

# Update NPC state during gameplay
arc.update_npc_state(
    npc_id=npc_result['npc_id'],
    new_motion_id=2,  # Transition to different motion
    blend_state=b'\x00' * 32,  # Blend state hash
    energy_level=75  # 75% energy
)

# Retrieve motion pack data
motion = arc.get_motion_pack(motion_id=1)
print(f"Creator: {motion['creator']}")
print(f"Styles: {motion['style_labels']}")
print(f"Usage Count: {motion['usage_count']}")

# Retrieve NPC state
npc_state = arc.get_npc_state(npc_id=1)
print(f"Current Motion: {npc_state['current_motion_id']}")
print(f"Energy: {npc_state['energy_level']}")
```

### Integration with Trustless Agent Loop

```python
from kinetic_ledger.services import TrustlessAgentLoop, ArcNetworkService
from kinetic_ledger.utils.canonicalize import compute_pack_hash

# Initialize services
agent = TrustlessAgentLoop(config=agent_config)
arc = ArcNetworkService()

# Execute blend workflow
result = agent.execute_blend_workflow(
    upload_request=upload,
    blend_request=blend,
    payment_proof="x402_proof_...",
    creator_address="0x..."
)

# If decision is MINT, mint on Arc
if result.decision == "MINT":
    pack_hash = compute_pack_hash(result.canonical_pack)
    
    arc_result = arc.mint_motion_pack(
        pack_hash=pack_hash,
        creator_address=creator_address,
        ipfs_uri=f"ipfs://{result.canonical_pack.ipfs_cid}",
        style_labels=result.canonical_pack.style_labels,
        npc_tags=result.canonical_pack.npc_tags
    )
    
    print(f"✅ Minted on Arc: Motion ID {arc_result['motion_id']}")
    print(f"   TX: {arc_result['tx_hash']}")
    print(f"   Explorer: https://testnet.arcscan.app/tx/{arc_result['tx_hash']}")
```

## Game Engine Integration

### Unity Example (kijani-spiral)

```csharp
using UnityEngine;
using Nethereum.Web3;
using Nethereum.Contracts;
using System.Threading.Tasks;

public class ArcNPCManager : MonoBehaviour
{
    private Web3 web3;
    private Contract npcMotionRegistry;
    
    private const string RPC_URL = "https://rpc.testnet.arc.network";
    private const string CONTRACT_ADDRESS = "0x32368037b14819C9e5Dbe96b3d67C59b8c65c4BF";
    
    async void Start()
    {
        web3 = new Web3(RPC_URL);
        npcMotionRegistry = web3.Eth.GetContract(ABI, CONTRACT_ADDRESS);
        
        // Spawn NPC with motion ID 1
        int npcId = await SpawnNPC(motionId: 1);
        Debug.Log($"NPC spawned with ID: {npcId}");
    }
    
    public async Task<int> SpawnNPC(int motionId)
    {
        var spawnFunction = npcMotionRegistry.GetFunction("spawnNPC");
        var receipt = await spawnFunction.SendTransactionAndWaitForReceiptAsync(
            from: playerWalletAddress,
            gas: new HexBigInteger(200000),
            value: null,
            functionInput: new object[] { motionId }
        );
        
        // Parse NPC ID from logs
        var npcSpawnedEvent = receipt.Logs[0];
        int npcId = (int)npcSpawnedEvent.Topics[1];
        
        return npcId;
    }
    
    public async Task UpdateNPCMotion(int npcId, int newMotionId, int energyLevel)
    {
        byte[] blendState = ComputeBlendStateHash(npcId, newMotionId);
        
        var updateFunction = npcMotionRegistry.GetFunction("updateNPCState");
        await updateFunction.SendTransactionAndWaitForReceiptAsync(
            from: playerWalletAddress,
            gas: new HexBigInteger(150000),
            value: null,
            functionInput: new object[] { npcId, newMotionId, blendState, energyLevel }
        );
        
        Debug.Log($"NPC {npcId} updated to motion {newMotionId}");
    }
}
```

## Payment Economics

### USDC Pricing

Arc uses USDC as the native gas token. Example pricing:

```python
# 1 USDC = 1,000,000 (6 decimals)
MINT_PRICE = 1_000_000  # 1 USDC per motion pack mint

# Usage fees (per second of animation)
USAGE_FEE_PER_SECOND = 100  # 0.0001 USDC per second

# Example: 10-second animation loop
duration = 10  # seconds
total_fee = USAGE_FEE_PER_SECOND * duration  # 1,000 = 0.001 USDC

# Royalty distribution:
# Creator:  0.001 * 0.70 = 0.0007 USDC
# Oracle:   0.001 * 0.10 = 0.0001 USDC
# Platform: 0.001 * 0.15 = 0.00015 USDC
# Ops:      0.001 * 0.05 = 0.00005 USDC
```

### Gas Costs (Estimated)

| Operation | Gas Estimate | USDC Cost (at 10 gwei) |
|-----------|--------------|------------------------|
| Mint Motion Pack | ~300,000 | ~$0.003 |
| Spawn NPC | ~150,000 | ~$0.0015 |
| Update NPC State | ~100,000 | ~$0.001 |
| Record Usage | ~80,000 | ~$0.0008 |

## Testing

### Run Foundry Tests

```bash
# Compile contracts
forge build

# Run tests
forge test -vv

# Run specific test
forge test --match-test testMintMotionPack -vvv
```

### Python Integration Tests

```bash
# Set environment
export ARC_TESTNET_RPC_URL="https://rpc.testnet.arc.network"
export NPC_MOTION_REGISTRY_ADDRESS="0x..."
export PRIVATE_KEY="0x..."

# Run tests
pytest tests/test_arc_integration.py -v
```

## Monitoring & Analytics

### Arc Explorer

View all transactions and contract interactions:
- **Testnet**: https://testnet.arcscan.app/address/0x32368037b14819C9e5Dbe96b3d67C59b8c65c4BF

### On-Chain Queries

```python
# Get motion pack stats
motion = arc.get_motion_pack(motion_id=1)
print(f"Usage Count: {motion['usage_count']}")
print(f"Creator: {motion['creator']}")
print(f"Active: {motion['is_active']}")

# Get NPC state
npc = arc.get_npc_state(npc_id=1)
print(f"Current Motion: {npc['current_motion_id']}")
print(f"Energy Level: {npc['energy_level']}")
print(f"Last Update Block: {npc['last_update_block']}")
```

## Security Best Practices

1. **Private Key Management**
   - Never commit `.env` files with real keys
   - Use hardware wallets for mainnet
   - Rotate keys periodically

2. **Contract Verification**
   - Verify contracts on Arc Explorer
   - Audit smart contracts before mainnet
   - Use multisig for admin functions

3. **Access Control**
   - Limit oracle authorization
   - Whitelist game engines
   - Monitor unauthorized access attempts

4. **Payment Validation**
   - Verify payment amounts before transactions
   - Check balance before sending
   - Handle failed transactions gracefully

## Troubleshooting

### Common Issues

**1. Transaction fails with "insufficient funds"**
```bash
# Check balance
cast balance $YOUR_ADDRESS --rpc-url $ARC_TESTNET_RPC_URL

# Get testnet USDC from faucet
# Visit https://faucet.circle.com
```

**2. Contract call reverts with "Only authorized oracle"**
```bash
# Verify oracle authorization
cast call $NPC_MOTION_REGISTRY_ADDRESS \
  "authorizedOracles(address)(bool)" \
  $ORACLE_ADDRESS \
  --rpc-url $ARC_TESTNET_RPC_URL
```

**3. NPC state update fails**
```bash
# Check if NPC exists and you're the owner
cast call $NPC_MOTION_REGISTRY_ADDRESS \
  "getNPCState(uint256)" \
  $NPC_ID \
  --rpc-url $ARC_TESTNET_RPC_URL
```

## Resources

- **Arc Documentation**: https://docs.arc.network
- **Arc Faucet**: https://faucet.circle.com
- **Arc Explorer**: https://testnet.arcscan.app
- **Foundry Book**: https://book.getfoundry.sh
- **Kijani Spiral (Unity)**: https://github.com/RydlrCS/kijani-spiral
- **Kinetic Ledger**: Current repository

## Next Steps

1. **Deploy to Arc Testnet** using deployment scripts
2. **Integrate with kijani-spiral** Unity game engine
3. **Test NPC spawning** and animation state updates
4. **Monitor usage metrics** via Arc Explorer
5. **Optimize gas costs** through batch operations
6. **Prepare for mainnet** deployment with audits

---

**Status**: ✅ Ready for Arc Testnet deployment
**Last Updated**: January 9, 2026
