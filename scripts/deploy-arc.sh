#!/bin/bash
# Deploy NPCMotionRegistry to Arc Testnet

set -e

# Load environment variables
source .env

echo "🚀 Deploying NPCMotionRegistry to Arc Testnet..."
echo ""
echo "Configuration:"
echo "  RPC URL: $ARC_TESTNET_RPC_URL"
echo "  Platform Address: $PLATFORM_ADDRESS"
echo "  Ops Address: $OPS_ADDRESS"
echo "  Mint Price: $MINT_PRICE USDC"
echo "  Usage Fee: $USAGE_FEE_PER_SECOND USDC/second"
echo ""

# Deploy contract
forge create contracts/NPCMotionRegistry.sol:NPCMotionRegistry \
  --rpc-url $ARC_TESTNET_RPC_URL \
  --private-key $PRIVATE_KEY \
  --constructor-args \
    "$PLATFORM_ADDRESS" \
    "$OPS_ADDRESS" \
    "$MINT_PRICE" \
    "$USAGE_FEE_PER_SECOND" \
  --broadcast \
  --verify

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Copy the 'Deployed to:' address to .env as NPC_MOTION_REGISTRY_ADDRESS"
echo "2. Verify on Arc Explorer: https://testnet.arcscan.app"
echo "3. Authorize oracle and game engines using scripts/authorize.sh"
