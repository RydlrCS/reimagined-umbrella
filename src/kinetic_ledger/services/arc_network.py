#!/usr/bin/env python3
"""
Arc Network Integration Service

Manages interaction with NPCMotionRegistry smart contract on Arc Testnet.
Handles motion pack minting, NPC spawning, and usage tracking.
"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.signers.local import LocalAccount

from ..utils.logging import get_logger
from ..utils.errors import E_DEP_ARC, E_CFG_MISSING

logger = get_logger(__name__)


class ArcNetworkConfig(BaseModel):
    """Configuration for Arc Network service."""
    rpc_url: str
    contract_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    private_key: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")
    chain_id: int = 52085143  # Arc testnet
    gas_price_gwei: Optional[float] = None


class ArcNetworkService:
    """
    Service for interacting with Arc Network smart contracts.
    
    Provides methods to:
    - Mint motion packs on-chain
    - Spawn NPCs with motion data
    - Update NPC animation state
    - Record usage and distribute payments
    """
    
    # Contract ABI (essential functions)
    CONTRACT_ABI = [
        {
            "inputs": [
                {"internalType": "bytes32", "name": "packHash", "type": "bytes32"},
                {"internalType": "address", "name": "creator", "type": "address"},
                {"internalType": "string", "name": "ipfsUri", "type": "string"},
                {"internalType": "string[]", "name": "styleLabels", "type": "string[]"},
                {"internalType": "string[]", "name": "npcTags", "type": "string[]"}
            ],
            "name": "mintMotionPack",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "uint256", "name": "motionId", "type": "uint256"}],
            "name": "spawnNPC",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "npcId", "type": "uint256"},
                {"internalType": "uint256", "name": "newMotionId", "type": "uint256"},
                {"internalType": "bytes32", "name": "blendState", "type": "bytes32"},
                {"internalType": "uint256", "name": "energyLevel", "type": "uint256"}
            ],
            "name": "updateNPCState",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "motionId", "type": "uint256"},
                {"internalType": "uint256", "name": "npcId", "type": "uint256"},
                {"internalType": "uint256", "name": "durationSeconds", "type": "uint256"}
            ],
            "name": "recordUsageAndPay",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "uint256", "name": "motionId", "type": "uint256"}],
            "name": "getMotionPack",
            "outputs": [
                {"internalType": "bytes32", "name": "packHash", "type": "bytes32"},
                {"internalType": "address", "name": "creator", "type": "address"},
                {"internalType": "uint256", "name": "mintedAt", "type": "uint256"},
                {"internalType": "string", "name": "ipfsUri", "type": "string"},
                {"internalType": "string[]", "name": "styleLabels", "type": "string[]"},
                {"internalType": "string[]", "name": "npcTags", "type": "string[]"},
                {"internalType": "bool", "name": "isActive", "type": "bool"},
                {"internalType": "uint256", "name": "usageCount", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "uint256", "name": "npcId", "type": "uint256"}],
            "name": "getNPCState",
            "outputs": [
                {"internalType": "uint256", "name": "currentMotionId", "type": "uint256"},
                {"internalType": "uint256", "name": "lastUpdateBlock", "type": "uint256"},
                {"internalType": "address", "name": "gameEngine", "type": "address"},
                {"internalType": "bytes32", "name": "blendState", "type": "bytes32"},
                {"internalType": "uint256", "name": "energyLevel", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    def __init__(
        self,
        config: Optional[ArcNetworkConfig] = None,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None
    ):
        """
        Initialize Arc Network service.
        
        Args:
            config: ArcNetworkConfig instance (preferred)
            rpc_url: Arc RPC endpoint (defaults to env var ARC_TESTNET_RPC_URL)
            private_key: Wallet private key (defaults to env var PRIVATE_KEY)
            contract_address: NPCMotionRegistry address (defaults to env var)
        """
        # Use config if provided, otherwise fall back to legacy params
        if config:
            self.rpc_url = config.rpc_url
            self.private_key = config.private_key
            self.contract_address = config.contract_address
            self.chain_id = config.chain_id
        else:
            self.rpc_url = rpc_url or os.getenv("ARC_TESTNET_RPC_URL")
            self.private_key = private_key or os.getenv("PRIVATE_KEY")
            self.contract_address = contract_address or os.getenv("NPC_MOTION_REGISTRY_ADDRESS")
            self.chain_id = 52085143  # Arc testnet default
        
        if not self.rpc_url:
            raise ValueError("ARC_TESTNET_RPC_URL not configured")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise ValueError(f"Failed to connect to Arc at {self.rpc_url}")
        
        logger.info(f"Connected to Arc Network: {self.rpc_url}")
        logger.info(f"Chain ID: {self.w3.eth.chain_id}")
        
        # Initialize account if private key provided
        self.account: Optional[LocalAccount] = None
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            logger.info(f"Wallet: {self.account.address}")
        
        # Initialize contract if address provided
        self.contract: Optional[Contract] = None
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=self.CONTRACT_ABI
            )
            logger.info(f"Contract: {self.contract_address}")
    
    def mint_motion_pack(
        self,
        pack_hash: bytes,
        creator_address: str,
        ipfs_uri: str,
        style_labels: List[str],
        npc_tags: List[str]
    ) -> Dict[str, Any]:
        """
        Mint a new motion pack on Arc Network.
        
        Args:
            pack_hash: Canonical pack hash (32 bytes from keccak256)
            creator_address: Motion creator's wallet address
            ipfs_uri: IPFS URI for BVH/FBX data
            style_labels: Style tags (capoeira, breakdance, etc.)
            npc_tags: NPC behavior tags (agile, evasive, etc.)
        
        Returns:
            Dict with transaction hash, motion ID, and gas used
        """
        if not self.contract or not self.account:
            raise E_CFG_MISSING("Contract or account not initialized")
        
        logger.info(f"Minting motion pack for creator {creator_address}")
        logger.info(f"  IPFS URI: {ipfs_uri}")
        logger.info(f"  Styles: {style_labels}")
        logger.info(f"  NPC Tags: {npc_tags}")
        
        # Build transaction
        tx = self.contract.functions.mintMotionPack(
            pack_hash,
            Web3.to_checksum_address(creator_address),
            ipfs_uri,
            style_labels,
            npc_tags
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Parse motion ID from logs
        motion_id = None
        for log in receipt['logs']:
            try:
                parsed = self.contract.events.MotionMinted().process_log(log)
                motion_id = parsed['args']['motionId']
                break
            except:
                continue
        
        logger.info(f"✅ Motion pack minted: ID={motion_id}, Gas={receipt['gasUsed']}")
        
        return {
            "tx_hash": tx_hash.hex(),
            "motion_id": motion_id,
            "gas_used": receipt['gasUsed'],
            "block_number": receipt['blockNumber']
        }
    
    def spawn_npc(self, motion_id: int) -> Dict[str, Any]:
        """
        Spawn a new NPC with initial motion.
        
        Args:
            motion_id: Motion pack ID to assign to NPC
        
        Returns:
            Dict with transaction hash, NPC ID, and gas used
        """
        if not self.contract or not self.account:
            raise E_CFG_MISSING("Contract or account not initialized")
        
        logger.info(f"Spawning NPC with motion ID {motion_id}")
        
        # Build transaction
        tx = self.contract.functions.spawnNPC(motion_id).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Parse NPC ID from logs
        npc_id = None
        for log in receipt['logs']:
            try:
                parsed = self.contract.events.NPCSpawned().process_log(log)
                npc_id = parsed['args']['npcId']
                break
            except:
                continue
        
        logger.info(f"✅ NPC spawned: ID={npc_id}, Motion={motion_id}")
        
        return {
            "tx_hash": tx_hash.hex(),
            "npc_id": npc_id,
            "motion_id": motion_id,
            "gas_used": receipt['gasUsed']
        }
    
    def update_npc_state(
        self,
        npc_id: int,
        new_motion_id: int,
        blend_state: bytes,
        energy_level: int
    ) -> Dict[str, Any]:
        """
        Update NPC animation state.
        
        Args:
            npc_id: NPC identifier
            new_motion_id: New motion pack to blend to
            blend_state: Animation blend state hash (32 bytes)
            energy_level: Energy level (0-100)
        
        Returns:
            Dict with transaction hash and gas used
        """
        if not self.contract or not self.account:
            raise E_CFG_MISSING("Contract or account not initialized")
        
        logger.info(f"Updating NPC {npc_id} state: motion={new_motion_id}, energy={energy_level}")
        
        tx = self.contract.functions.updateNPCState(
            npc_id,
            new_motion_id,
            blend_state,
            energy_level
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 150000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        logger.info(f"✅ NPC state updated: {tx_hash.hex()}")
        
        return {
            "tx_hash": tx_hash.hex(),
            "gas_used": receipt['gasUsed']
        }
    
    def get_motion_pack(self, motion_id: int) -> Dict[str, Any]:
        """
        Retrieve motion pack data from blockchain.
        
        Args:
            motion_id: Motion pack ID
        
        Returns:
            Dict with motion pack details
        """
        if not self.contract:
            raise E_CFG_MISSING("Contract not initialized")
        
        result = self.contract.functions.getMotionPack(motion_id).call()
        
        return {
            "motion_id": motion_id,
            "pack_hash": result[0].hex(),
            "creator": result[1],
            "minted_at": datetime.fromtimestamp(result[2]),
            "ipfs_uri": result[3],
            "style_labels": result[4],
            "npc_tags": result[5],
            "is_active": result[6],
            "usage_count": result[7]
        }
    
    def get_npc_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Retrieve NPC state from blockchain.
        
        Args:
            npc_id: NPC identifier
        
        Returns:
            Dict with NPC state details
        """
        if not self.contract:
            raise E_CFG_MISSING("Contract not initialized")
        
        result = self.contract.functions.getNPCState(npc_id).call()
        
        return {
            "npc_id": npc_id,
            "current_motion_id": result[0],
            "last_update_block": result[1],
            "game_engine": result[2],
            "blend_state": result[3].hex(),
            "energy_level": result[4]
        }
