// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title NPCMotionRegistry
 * @notice Manages on-chain NPC animation state and motion pack ownership for game engines
 * @dev Integrates with Kinetic Ledger attestation oracle for motion novelty validation
 */
contract NPCMotionRegistry {
    // ========== STATE VARIABLES ==========
    
    struct MotionPack {
        bytes32 packHash;           // keccak256 of canonical pack (from attestation oracle)
        address creator;            // Motion creator address
        uint256 mintedAt;           // Block timestamp when minted
        string ipfsUri;             // IPFS URI for BVH/FBX data
        string[] styleLabels;       // Style tags (capoeira, breakdance, etc.)
        string[] npcTags;           // NPC behavior tags (agile, evasive, etc.)
        bool isActive;              // Whether motion is approved for use
        uint256 usageCount;         // Number of times used in-game
    }
    
    struct NPCState {
        uint256 currentMotionId;    // Active motion pack ID
        uint256 lastUpdateBlock;    // Last state update
        address gameEngine;         // Authorized game engine contract
        bytes32 blendState;         // Current animation blend state hash
        uint256 energyLevel;        // NPC energy/stamina (0-100)
    }
    
    // Motion pack storage
    mapping(uint256 => MotionPack) public motionPacks;
    uint256 public nextMotionId;
    
    // NPC state tracking
    mapping(uint256 => NPCState) public npcStates;  // npcId => state
    uint256 public nextNpcId;
    
    // Access control
    mapping(address => bool) public authorizedOracles;
    mapping(address => bool) public authorizedEngines;
    address public owner;
    
    // Pricing (in USDC - Arc's native gas token)
    uint256 public mintPrice;           // Cost to mint new motion pack
    uint256 public usageFeePerSecond;   // Per-second usage fee
    
    // Revenue splits
    struct PayoutSplit {
        uint256 creator;    // 70% to creator
        uint256 oracle;     // 10% to oracle
        uint256 platform;   // 15% to platform
        uint256 ops;        // 5% to ops
    }
    PayoutSplit public payoutSplit;
    
    address public platformAddress;
    address public opsAddress;
    
    // ========== EVENTS ==========
    
    event MotionMinted(
        uint256 indexed motionId,
        bytes32 indexed packHash,
        address indexed creator,
        string ipfsUri
    );
    
    event NPCSpawned(
        uint256 indexed npcId,
        uint256 indexed motionId,
        address indexed gameEngine
    );
    
    event NPCStateUpdated(
        uint256 indexed npcId,
        uint256 newMotionId,
        bytes32 blendState,
        uint256 energyLevel
    );
    
    event MotionUsageRecorded(
        uint256 indexed motionId,
        uint256 indexed npcId,
        uint256 durationSeconds,
        uint256 feePaid
    );
    
    event PayoutDistributed(
        uint256 indexed motionId,
        address creator,
        uint256 creatorAmount,
        uint256 oracleAmount,
        uint256 platformAmount,
        uint256 opsAmount
    );
    
    // ========== MODIFIERS ==========
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "Only authorized oracle");
        _;
    }
    
    modifier onlyAuthorizedEngine() {
        require(authorizedEngines[msg.sender], "Only authorized game engine");
        _;
    }
    
    // ========== CONSTRUCTOR ==========
    
    constructor(
        address _platformAddress,
        address _opsAddress,
        uint256 _mintPrice,
        uint256 _usageFeePerSecond
    ) {
        owner = msg.sender;
        platformAddress = _platformAddress;
        opsAddress = _opsAddress;
        mintPrice = _mintPrice;
        usageFeePerSecond = _usageFeePerSecond;
        
        // Initialize payout splits (must sum to 10000 = 100%)
        payoutSplit = PayoutSplit({
            creator: 7000,   // 70%
            oracle: 1000,    // 10%
            platform: 1500,  // 15%
            ops: 500         // 5%
        });
        
        nextMotionId = 1;
        nextNpcId = 1;
    }
    
    // ========== ORACLE FUNCTIONS ==========
    
    /**
     * @notice Mint a new motion pack (called by attestation oracle after validation)
     * @param packHash Canonical pack hash from oracle
     * @param creator Motion creator address
     * @param ipfsUri IPFS URI for motion data
     * @param styleLabels Style tags array
     * @param npcTags NPC behavior tags array
     */
    function mintMotionPack(
        bytes32 packHash,
        address creator,
        string memory ipfsUri,
        string[] memory styleLabels,
        string[] memory npcTags
    ) external onlyAuthorizedOracle returns (uint256) {
        uint256 motionId = nextMotionId++;
        
        motionPacks[motionId] = MotionPack({
            packHash: packHash,
            creator: creator,
            mintedAt: block.timestamp,
            ipfsUri: ipfsUri,
            styleLabels: styleLabels,
            npcTags: npcTags,
            isActive: true,
            usageCount: 0
        });
        
        emit MotionMinted(motionId, packHash, creator, ipfsUri);
        
        return motionId;
    }
    
    // ========== GAME ENGINE FUNCTIONS ==========
    
    /**
     * @notice Spawn a new NPC with initial motion
     * @param motionId Initial motion pack ID
     */
    function spawnNPC(uint256 motionId) external onlyAuthorizedEngine returns (uint256) {
        require(motionPacks[motionId].isActive, "Motion not active");
        
        uint256 npcId = nextNpcId++;
        
        npcStates[npcId] = NPCState({
            currentMotionId: motionId,
            lastUpdateBlock: block.number,
            gameEngine: msg.sender,
            blendState: keccak256(abi.encodePacked("initial", motionId)),
            energyLevel: 100
        });
        
        emit NPCSpawned(npcId, motionId, msg.sender);
        
        return npcId;
    }
    
    /**
     * @notice Update NPC animation state
     * @param npcId NPC identifier
     * @param newMotionId New motion pack to blend to
     * @param blendState Animation blend state hash
     * @param energyLevel Updated energy level
     */
    function updateNPCState(
        uint256 npcId,
        uint256 newMotionId,
        bytes32 blendState,
        uint256 energyLevel
    ) external onlyAuthorizedEngine {
        require(npcStates[npcId].gameEngine == msg.sender, "Not NPC owner");
        require(motionPacks[newMotionId].isActive, "Motion not active");
        require(energyLevel <= 100, "Energy must be <= 100");
        
        npcStates[npcId].currentMotionId = newMotionId;
        npcStates[npcId].lastUpdateBlock = block.number;
        npcStates[npcId].blendState = blendState;
        npcStates[npcId].energyLevel = energyLevel;
        
        emit NPCStateUpdated(npcId, newMotionId, blendState, energyLevel);
    }
    
    /**
     * @notice Record motion usage and distribute payments
     * @param motionId Motion pack used
     * @param npcId NPC that used the motion
     * @param durationSeconds How long the motion was used
     */
    function recordUsageAndPay(
        uint256 motionId,
        uint256 npcId,
        uint256 durationSeconds
    ) external payable onlyAuthorizedEngine {
        MotionPack storage pack = motionPacks[motionId];
        require(pack.isActive, "Motion not active");
        
        // Calculate fee (in USDC wei - Arc native token)
        uint256 totalFee = usageFeePerSecond * durationSeconds;
        require(msg.value >= totalFee, "Insufficient payment");
        
        // Distribute payments according to splits
        uint256 creatorAmount = (totalFee * payoutSplit.creator) / 10000;
        uint256 oracleAmount = (totalFee * payoutSplit.oracle) / 10000;
        uint256 platformAmount = (totalFee * payoutSplit.platform) / 10000;
        uint256 opsAmount = (totalFee * payoutSplit.ops) / 10000;
        
        // Transfer payments
        payable(pack.creator).transfer(creatorAmount);
        payable(msg.sender).transfer(oracleAmount);  // Oracle is the game engine in this case
        payable(platformAddress).transfer(platformAmount);
        payable(opsAddress).transfer(opsAmount);
        
        // Update usage count
        pack.usageCount++;
        
        emit MotionUsageRecorded(motionId, npcId, durationSeconds, totalFee);
        emit PayoutDistributed(
            motionId,
            pack.creator,
            creatorAmount,
            oracleAmount,
            platformAmount,
            opsAmount
        );
        
        // Refund excess payment
        if (msg.value > totalFee) {
            payable(msg.sender).transfer(msg.value - totalFee);
        }
    }
    
    // ========== VIEW FUNCTIONS ==========
    
    function getMotionPack(uint256 motionId) external view returns (
        bytes32 packHash,
        address creator,
        uint256 mintedAt,
        string memory ipfsUri,
        string[] memory styleLabels,
        string[] memory npcTags,
        bool isActive,
        uint256 usageCount
    ) {
        MotionPack memory pack = motionPacks[motionId];
        return (
            pack.packHash,
            pack.creator,
            pack.mintedAt,
            pack.ipfsUri,
            pack.styleLabels,
            pack.npcTags,
            pack.isActive,
            pack.usageCount
        );
    }
    
    function getNPCState(uint256 npcId) external view returns (
        uint256 currentMotionId,
        uint256 lastUpdateBlock,
        address gameEngine,
        bytes32 blendState,
        uint256 energyLevel
    ) {
        NPCState memory state = npcStates[npcId];
        return (
            state.currentMotionId,
            state.lastUpdateBlock,
            state.gameEngine,
            state.blendState,
            state.energyLevel
        );
    }
    
    // ========== ADMIN FUNCTIONS ==========
    
    function authorizeOracle(address oracle, bool authorized) external onlyOwner {
        authorizedOracles[oracle] = authorized;
    }
    
    function authorizeEngine(address engine, bool authorized) external onlyOwner {
        authorizedEngines[engine] = authorized;
    }
    
    function updatePricing(uint256 newMintPrice, uint256 newUsageFee) external onlyOwner {
        mintPrice = newMintPrice;
        usageFeePerSecond = newUsageFee;
    }
    
    function deactivateMotion(uint256 motionId) external onlyOwner {
        motionPacks[motionId].isActive = false;
    }
}
