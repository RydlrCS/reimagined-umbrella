#!/bin/bash
# Authorize oracle and game engines for NPCMotionRegistry

set -e

source .env

echo "🔐 Authorizing oracle and game engines..."
echo ""

# Authorize attestation oracle
echo "Authorizing oracle: $ORACLE_ADDRESS"
cast send $NPC_MOTION_REGISTRY_ADDRESS \
  "authorizeOracle(address,bool)" \
  $ORACLE_ADDRESS \
  true \
  --rpc-url $ARC_TESTNET_RPC_URL \
  --private-key $PRIVATE_KEY

# Authorize game engines (comma-separated list)
IFS=',' read -ra ENGINES <<< "$GAME_ENGINE_ADDRESSES"
for engine in "${ENGINES[@]}"; do
  echo "Authorizing game engine: $engine"
  cast send $NPC_MOTION_REGISTRY_ADDRESS \
    "authorizeEngine(address,bool)" \
    $engine \
    true \
    --rpc-url $ARC_TESTNET_RPC_URL \
    --private-key $PRIVATE_KEY
done

echo ""
echo "✅ Authorization complete!"
