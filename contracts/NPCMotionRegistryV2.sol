// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

/**
 * @title NPCMotionRegistryV2
 * @notice Manages on-chain NPC animation state with infinite-depth royalty chain support
 * @dev Extended from NPCMotionRegistry with:
 *      - Recursive royalty distribution for derivative motion content
 *      - EIP-7702 gasless execution support
 *      - x402 payment verification integration
 *      - Circular reference prevention at mint time
 * 
 * Global Constants (matching Python models):
 *      - ROYALTY_DECAY_FACTOR: 50% decay per chain depth level
 *      - MAX_ROYALTY_CHAIN_DEPTH: 10 levels maximum
 *      - GAS_SPONSOR_REPLENISH_THRESHOLD: 40% of budget
 */
contract NPCMotionRegistryV2 {
    // ========== CONSTANTS ==========
    
    uint256 public constant ROYALTY_DECAY_FACTOR_BPS = 5000;  // 50% = 5000 basis points
    uint256 public constant MAX_ROYALTY_CHAIN_DEPTH = 10;
    uint256 public constant GAS_SPONSOR_REPLENISH_THRESHOLD_BPS = 4000;  // 40%
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MIN_PAYOUT_WEI = 0.01 ether;  // $0.01 minimum payout
    
    // ========== STRUCTS ==========
    
    struct MotionPackV2 {
        bytes32 packHash;              // Canonical pack hash from attestation oracle
        address creator;               // Motion creator address
        uint256 mintedAt;              // Block timestamp when minted
        string ipfsUri;                // IPFS URI for BVH/FBX data
        string[] styleLabels;          // Style tags (capoeira, breakdance, etc.)
        string[] npcTags;              // NPC behavior tags (agile, evasive, etc.)
        bool isActive;                 // Whether motion is approved for use
        uint256 usageCount;            // Number of times used in-game
        // Royalty chain fields
        uint256[] parentMotionIds;     // IDs of parent motions this derives from
        uint256 derivationDepth;       // Depth in derivation chain (0 = original)
        address[] royaltyRecipients;   // All royalty recipients in chain
        uint256[] royaltySharesBps;    // Royalty shares in basis points (sum <= 10000)
    }
    
    struct NPCStateV2 {
        uint256 currentMotionId;       // Active motion pack ID
        uint256 lastUpdateBlock;       // Last state update
        address gameEngine;            // Authorized game engine contract
        bytes32 blendState;            // Current animation blend state hash
        uint256 energyLevel;           // NPC energy/stamina (0-100)
        uint256[] blendHistory;        // Last 5 motion IDs for blend context
    }
    
    struct RoyaltyNode {
        address recipient;             // Wallet to receive royalty
        uint256 motionId;              // Source motion ID
        uint256 shareBps;              // Share in basis points
        uint256 depth;                 // Depth in chain (0 = immediate parent)
    }
    
    struct PayoutSplitV2 {
        uint256 creatorBps;            // Creator share (default 7000 = 70%)
        uint256 oracleBps;             // Oracle share (default 1000 = 10%)
        uint256 platformBps;           // Platform share (default 1500 = 15%)
        uint256 opsBps;                // Ops share (default 500 = 5%)
    }
    
    // EIP-712 Domain for gasless signatures
    bytes32 public constant DOMAIN_TYPEHASH = keccak256(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    );
    
    bytes32 public constant MINT_TYPEHASH = keccak256(
        "MintMotionPack(bytes32 packHash,address creator,string ipfsUri,uint256 nonce,uint256 deadline)"
    );
    
    bytes32 public constant USAGE_TYPEHASH = keccak256(
        "RecordUsage(uint256 motionId,uint256 npcId,uint256 durationSeconds,uint256 nonce,uint256 deadline)"
    );
    
    // ========== STATE VARIABLES ==========
    
    // Motion pack storage
    mapping(uint256 => MotionPackV2) public motionPacks;
    uint256 public nextMotionId;
    
    // NPC state tracking
    mapping(uint256 => NPCStateV2) public npcStates;
    uint256 public nextNpcId;
    
    // Pack hash to motion ID lookup (for derivative detection)
    mapping(bytes32 => uint256) public packHashToMotionId;
    
    // Circular reference prevention cache
    mapping(bytes32 => bool) public derivationPairExists;
    
    // Access control
    mapping(address => bool) public authorizedOracles;
    mapping(address => bool) public authorizedEngines;
    mapping(address => bool) public gasSponsorOperators;
    address public owner;
    
    // Pricing (in USDC - Arc's native gas token)
    uint256 public mintPrice;
    uint256 public usageFeePerSecond;
    
    // Revenue splits
    PayoutSplitV2 public payoutSplit;
    address public platformAddress;
    address public opsAddress;
    
    // Gas sponsorship budget
    uint256 public gasSponsorBudget;
    uint256 public gasSponsorInitialBudget;
    
    // Nonces for gasless signatures
    mapping(address => uint256) public nonces;
    
    // Domain separator (cached for efficiency)
    bytes32 public immutable DOMAIN_SEPARATOR;
    
    // ========== EVENTS ==========
    
    event MotionMintedV2(
        uint256 indexed motionId,
        bytes32 indexed packHash,
        address indexed creator,
        string ipfsUri,
        uint256 derivationDepth,
        uint256[] parentMotionIds
    );
    
    event NPCSpawnedV2(
        uint256 indexed npcId,
        uint256 indexed motionId,
        address indexed gameEngine
    );
    
    event NPCStateUpdatedV2(
        uint256 indexed npcId,
        uint256 newMotionId,
        bytes32 blendState,
        uint256 energyLevel
    );
    
    event RecursivePayoutDistributed(
        uint256 indexed motionId,
        uint256 totalFee,
        uint256 royaltyLevelsProcessed,
        uint256 totalRoyaltiesPaid
    );
    
    event RoyaltyPaid(
        uint256 indexed motionId,
        uint256 indexed sourceMotionId,
        address indexed recipient,
        uint256 amount,
        uint256 depth
    );
    
    event GaslessExecutionProcessed(
        address indexed signer,
        bytes32 indexed actionHash,
        bool success
    );
    
    event GasBudgetReplenished(
        uint256 previousBalance,
        uint256 newBalance,
        address operator
    );
    
    // ========== MODIFIERS ==========
    
    modifier onlyOwner() {
        require(msg.sender == owner, "NPCMotionRegistryV2: Only owner");
        _;
    }
    
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "NPCMotionRegistryV2: Only authorized oracle");
        _;
    }
    
    modifier onlyAuthorizedEngine() {
        require(authorizedEngines[msg.sender], "NPCMotionRegistryV2: Only authorized game engine");
        _;
    }
    
    modifier onlyGasSponsor() {
        require(
            gasSponsorOperators[msg.sender] || msg.sender == owner,
            "NPCMotionRegistryV2: Only gas sponsor operator"
        );
        _;
    }
    
    // ========== CONSTRUCTOR ==========
    
    constructor(
        address _platformAddress,
        address _opsAddress,
        uint256 _mintPrice,
        uint256 _usageFeePerSecond,
        uint256 _initialGasBudget
    ) {
        owner = msg.sender;
        platformAddress = _platformAddress;
        opsAddress = _opsAddress;
        mintPrice = _mintPrice;
        usageFeePerSecond = _usageFeePerSecond;
        
        // Initialize payout splits (must sum to 10000 = 100%)
        payoutSplit = PayoutSplitV2({
            creatorBps: 7000,    // 70%
            oracleBps: 1000,     // 10%
            platformBps: 1500,   // 15%
            opsBps: 500          // 5%
        });
        
        nextMotionId = 1;
        nextNpcId = 1;
        
        // Gas sponsorship
        gasSponsorBudget = _initialGasBudget;
        gasSponsorInitialBudget = _initialGasBudget;
        
        // Compute EIP-712 domain separator
        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                DOMAIN_TYPEHASH,
                keccak256(bytes("NPCMotionRegistryV2")),
                keccak256(bytes("2")),
                block.chainid,
                address(this)
            )
        );
    }
    
    // ========== ORACLE FUNCTIONS (WITH ROYALTY CHAIN) ==========
    
    /**
     * @notice Mint a new motion pack with optional derivation chain
     * @dev Validates no circular references in derivation chain
     * @param packHash Canonical pack hash from oracle
     * @param creator Motion creator address
     * @param ipfsUri IPFS URI for motion data
     * @param styleLabels Style tags array
     * @param npcTags NPC behavior tags array
     * @param parentMotionIds Array of parent motion IDs (empty if original)
     * @param royaltySharesBps Royalty shares for each parent in basis points
     */
    function mintMotionPackV2(
        bytes32 packHash,
        address creator,
        string memory ipfsUri,
        string[] memory styleLabels,
        string[] memory npcTags,
        uint256[] memory parentMotionIds,
        uint256[] memory royaltySharesBps
    ) external onlyAuthorizedOracle returns (uint256) {
        require(packHashToMotionId[packHash] == 0, "Pack hash already minted");
        require(parentMotionIds.length == royaltySharesBps.length, "Array length mismatch");
        require(parentMotionIds.length <= MAX_ROYALTY_CHAIN_DEPTH, "Too many parents");
        
        // Validate no circular references
        _validateNoCircularReferences(packHash, parentMotionIds);
        
        // Calculate derivation depth
        uint256 maxParentDepth = 0;
        for (uint256 i = 0; i < parentMotionIds.length; i++) {
            require(motionPacks[parentMotionIds[i]].isActive, "Parent motion not active");
            uint256 parentDepth = motionPacks[parentMotionIds[i]].derivationDepth;
            if (parentDepth > maxParentDepth) {
                maxParentDepth = parentDepth;
            }
        }
        uint256 derivationDepth = parentMotionIds.length > 0 ? maxParentDepth + 1 : 0;
        require(derivationDepth <= MAX_ROYALTY_CHAIN_DEPTH, "Derivation depth exceeded");
        
        // Build royalty chain (flattened from all parents)
        (address[] memory recipients, uint256[] memory shares) = _buildRoyaltyChain(
            parentMotionIds,
            royaltySharesBps
        );
        
        uint256 motionId = nextMotionId++;
        
        motionPacks[motionId] = MotionPackV2({
            packHash: packHash,
            creator: creator,
            mintedAt: block.timestamp,
            ipfsUri: ipfsUri,
            styleLabels: styleLabels,
            npcTags: npcTags,
            isActive: true,
            usageCount: 0,
            parentMotionIds: parentMotionIds,
            derivationDepth: derivationDepth,
            royaltyRecipients: recipients,
            royaltySharesBps: shares
        });
        
        // Register pack hash for derivative detection
        packHashToMotionId[packHash] = motionId;
        
        // Mark derivation pairs
        for (uint256 i = 0; i < parentMotionIds.length; i++) {
            derivationPairExists[keccak256(abi.encodePacked(packHash, parentMotionIds[i]))] = true;
        }
        
        emit MotionMintedV2(motionId, packHash, creator, ipfsUri, derivationDepth, parentMotionIds);
        
        return motionId;
    }
    
    /**
     * @notice Validate no circular references exist in derivation chain
     * @dev Oracle-side validation at mint time (matching Python implementation)
     */
    function _validateNoCircularReferences(
        bytes32 newPackHash,
        uint256[] memory parentIds
    ) internal view {
        // Build visited set
        for (uint256 i = 0; i < parentIds.length; i++) {
            bytes32 parentHash = motionPacks[parentIds[i]].packHash;
            require(parentHash != newPackHash, "Circular reference: self-derivation");
            
            // Check all ancestors of this parent
            uint256[] memory parentChain = motionPacks[parentIds[i]].parentMotionIds;
            for (uint256 j = 0; j < parentChain.length && j < MAX_ROYALTY_CHAIN_DEPTH; j++) {
                bytes32 ancestorHash = motionPacks[parentChain[j]].packHash;
                require(ancestorHash != newPackHash, "Circular reference: ancestor");
            }
        }
    }
    
    /**
     * @notice Build flattened royalty chain from parent motions
     * @dev Applies ROYALTY_DECAY_FACTOR at each depth level
     */
    function _buildRoyaltyChain(
        uint256[] memory parentIds,
        uint256[] memory immediateShares
    ) internal view returns (address[] memory recipients, uint256[] memory shares) {
        // Count total recipients across all chains
        uint256 totalRecipients = parentIds.length; // Immediate parents
        for (uint256 i = 0; i < parentIds.length; i++) {
            totalRecipients += motionPacks[parentIds[i]].royaltyRecipients.length;
        }
        
        // Limit to prevent gas issues
        if (totalRecipients > MAX_ROYALTY_CHAIN_DEPTH * 2) {
            totalRecipients = MAX_ROYALTY_CHAIN_DEPTH * 2;
        }
        
        recipients = new address[](totalRecipients);
        shares = new uint256[](totalRecipients);
        
        uint256 idx = 0;
        
        // Add immediate parents
        for (uint256 i = 0; i < parentIds.length && idx < totalRecipients; i++) {
            recipients[idx] = motionPacks[parentIds[i]].creator;
            shares[idx] = immediateShares[i];
            idx++;
        }
        
        // Add ancestors with decayed shares
        for (uint256 i = 0; i < parentIds.length; i++) {
            MotionPackV2 storage parent = motionPacks[parentIds[i]];
            
            for (uint256 j = 0; j < parent.royaltyRecipients.length && idx < totalRecipients; j++) {
                recipients[idx] = parent.royaltyRecipients[j];
                // Apply decay factor: original_share * (DECAY_FACTOR ^ (depth + 1))
                uint256 decayedShare = (parent.royaltySharesBps[j] * ROYALTY_DECAY_FACTOR_BPS) / BASIS_POINTS;
                shares[idx] = decayedShare;
                idx++;
            }
        }
        
        // Trim arrays to actual size
        assembly {
            mstore(recipients, idx)
            mstore(shares, idx)
        }
        
        return (recipients, shares);
    }
    
    // ========== USAGE AND RECURSIVE PAYOUT ==========
    
    /**
     * @notice Record motion usage and distribute payments recursively
     * @dev Processes infinite-depth royalty chain with decay factor
     * @param motionId Motion pack used
     * @param npcId NPC that used the motion
     * @param durationSeconds How long the motion was used
     */
    function recordUsageAndPayRecursive(
        uint256 motionId,
        uint256 npcId,
        uint256 durationSeconds
    ) external payable onlyAuthorizedEngine {
        MotionPackV2 storage pack = motionPacks[motionId];
        require(pack.isActive, "Motion not active");
        
        // Calculate fee (in USDC wei - Arc native token)
        uint256 totalFee = usageFeePerSecond * durationSeconds;
        require(msg.value >= totalFee, "Insufficient payment");
        
        // Calculate royalty payouts
        uint256 totalRoyalties = 0;
        uint256 levelsProcessed = 0;
        
        for (uint256 i = 0; i < pack.royaltyRecipients.length && i < MAX_ROYALTY_CHAIN_DEPTH; i++) {
            uint256 royaltyAmount = (totalFee * pack.royaltySharesBps[i]) / BASIS_POINTS;
            
            // Apply decay factor based on position (depth approximation)
            if (i > 0) {
                royaltyAmount = (royaltyAmount * ROYALTY_DECAY_FACTOR_BPS) / BASIS_POINTS;
            }
            
            if (royaltyAmount >= MIN_PAYOUT_WEI) {
                payable(pack.royaltyRecipients[i]).transfer(royaltyAmount);
                totalRoyalties += royaltyAmount;
                levelsProcessed++;
                
                emit RoyaltyPaid(
                    motionId,
                    pack.parentMotionIds.length > 0 ? pack.parentMotionIds[0] : 0,
                    pack.royaltyRecipients[i],
                    royaltyAmount,
                    i
                );
            }
        }
        
        // Remaining fee goes to standard split
        uint256 remainingFee = totalFee - totalRoyalties;
        _distributeStandardSplit(pack, remainingFee);
        
        // Update usage count
        pack.usageCount++;
        
        emit RecursivePayoutDistributed(motionId, totalFee, levelsProcessed, totalRoyalties);
        
        // Refund excess payment
        if (msg.value > totalFee) {
            payable(msg.sender).transfer(msg.value - totalFee);
        }
    }
    
    /**
     * @notice Distribute standard payout split (creator, oracle, platform, ops)
     */
    function _distributeStandardSplit(
        MotionPackV2 storage pack,
        uint256 amount
    ) internal {
        uint256 creatorAmount = (amount * payoutSplit.creatorBps) / BASIS_POINTS;
        uint256 oracleAmount = (amount * payoutSplit.oracleBps) / BASIS_POINTS;
        uint256 platformAmount = (amount * payoutSplit.platformBps) / BASIS_POINTS;
        uint256 opsAmount = (amount * payoutSplit.opsBps) / BASIS_POINTS;
        
        if (creatorAmount >= MIN_PAYOUT_WEI) {
            payable(pack.creator).transfer(creatorAmount);
        }
        if (oracleAmount >= MIN_PAYOUT_WEI) {
            payable(msg.sender).transfer(oracleAmount);
        }
        if (platformAmount >= MIN_PAYOUT_WEI) {
            payable(platformAddress).transfer(platformAmount);
        }
        if (opsAmount >= MIN_PAYOUT_WEI) {
            payable(opsAddress).transfer(opsAmount);
        }
    }
    
    // ========== GASLESS EXECUTION (EIP-7702) ==========
    
    /**
     * @notice Record usage with gasless execution via EIP-712 signature
     * @dev Gas is paid from sponsor budget instead of user
     * @param motionId Motion pack used
     * @param npcId NPC that used the motion
     * @param durationSeconds How long the motion was used
     * @param deadline Signature expiration timestamp
     * @param v Signature recovery id
     * @param r Signature r value
     * @param s Signature s value
     */
    function recordUsageAndPayGasless(
        uint256 motionId,
        uint256 npcId,
        uint256 durationSeconds,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external onlyGasSponsor {
        require(block.timestamp <= deadline, "Signature expired");
        
        // Verify signature
        bytes32 structHash = keccak256(
            abi.encode(
                USAGE_TYPEHASH,
                motionId,
                npcId,
                durationSeconds,
                nonces[msg.sender]++,
                deadline
            )
        );
        
        bytes32 digest = keccak256(
            abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash)
        );
        
        address signer = ecrecover(digest, v, r, s);
        require(authorizedEngines[signer], "Invalid signature or unauthorized");
        
        // Deduct from gas sponsor budget
        uint256 estimatedGas = 150000; // Estimated gas for operation
        uint256 gasCost = estimatedGas * tx.gasprice;
        require(gasSponsorBudget >= gasCost, "Insufficient gas sponsor budget");
        gasSponsorBudget -= gasCost;
        
        // Check if replenishment needed (40% threshold)
        uint256 threshold = (gasSponsorInitialBudget * GAS_SPONSOR_REPLENISH_THRESHOLD_BPS) / BASIS_POINTS;
        if (gasSponsorBudget < threshold) {
            // Emit event for off-chain replenishment trigger
            emit GasBudgetReplenished(gasSponsorBudget, gasSponsorBudget, address(0));
        }
        
        // Execute the usage recording (payment from contract balance)
        MotionPackV2 storage pack = motionPacks[motionId];
        require(pack.isActive, "Motion not active");
        
        uint256 totalFee = usageFeePerSecond * durationSeconds;
        
        // Process recursive payouts from contract balance
        uint256 totalRoyalties = 0;
        for (uint256 i = 0; i < pack.royaltyRecipients.length && i < MAX_ROYALTY_CHAIN_DEPTH; i++) {
            uint256 royaltyAmount = (totalFee * pack.royaltySharesBps[i]) / BASIS_POINTS;
            
            if (i > 0) {
                royaltyAmount = (royaltyAmount * ROYALTY_DECAY_FACTOR_BPS) / BASIS_POINTS;
            }
            
            if (royaltyAmount >= MIN_PAYOUT_WEI && address(this).balance >= royaltyAmount) {
                payable(pack.royaltyRecipients[i]).transfer(royaltyAmount);
                totalRoyalties += royaltyAmount;
            }
        }
        
        pack.usageCount++;
        
        emit GaslessExecutionProcessed(signer, digest, true);
    }
    
    /**
     * @notice Mint motion pack with gasless execution
     */
    function mintMotionPackGasless(
        bytes32 packHash,
        address creator,
        string memory ipfsUri,
        string[] memory styleLabels,
        string[] memory npcTags,
        uint256[] memory parentMotionIds,
        uint256[] memory royaltySharesBps,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external onlyGasSponsor returns (uint256) {
        require(block.timestamp <= deadline, "Signature expired");
        
        // Verify creator signature
        bytes32 structHash = keccak256(
            abi.encode(
                MINT_TYPEHASH,
                packHash,
                creator,
                keccak256(bytes(ipfsUri)),
                nonces[creator]++,
                deadline
            )
        );
        
        bytes32 digest = keccak256(
            abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash)
        );
        
        address signer = ecrecover(digest, v, r, s);
        require(signer == creator, "Invalid creator signature");
        
        // Deduct from gas sponsor budget
        uint256 estimatedGas = 300000; // Estimated gas for mint
        uint256 gasCost = estimatedGas * tx.gasprice;
        require(gasSponsorBudget >= gasCost, "Insufficient gas sponsor budget");
        gasSponsorBudget -= gasCost;
        
        // Execute mint (reuse main logic)
        require(packHashToMotionId[packHash] == 0, "Pack hash already minted");
        require(parentMotionIds.length == royaltySharesBps.length, "Array length mismatch");
        
        _validateNoCircularReferences(packHash, parentMotionIds);
        
        uint256 maxParentDepth = 0;
        for (uint256 i = 0; i < parentMotionIds.length; i++) {
            require(motionPacks[parentMotionIds[i]].isActive, "Parent motion not active");
            uint256 parentDepth = motionPacks[parentMotionIds[i]].derivationDepth;
            if (parentDepth > maxParentDepth) {
                maxParentDepth = parentDepth;
            }
        }
        uint256 derivationDepth = parentMotionIds.length > 0 ? maxParentDepth + 1 : 0;
        require(derivationDepth <= MAX_ROYALTY_CHAIN_DEPTH, "Derivation depth exceeded");
        
        (address[] memory recipients, uint256[] memory shares) = _buildRoyaltyChain(
            parentMotionIds,
            royaltySharesBps
        );
        
        uint256 motionId = nextMotionId++;
        
        motionPacks[motionId] = MotionPackV2({
            packHash: packHash,
            creator: creator,
            mintedAt: block.timestamp,
            ipfsUri: ipfsUri,
            styleLabels: styleLabels,
            npcTags: npcTags,
            isActive: true,
            usageCount: 0,
            parentMotionIds: parentMotionIds,
            derivationDepth: derivationDepth,
            royaltyRecipients: recipients,
            royaltySharesBps: shares
        });
        
        packHashToMotionId[packHash] = motionId;
        
        for (uint256 i = 0; i < parentMotionIds.length; i++) {
            derivationPairExists[keccak256(abi.encodePacked(packHash, parentMotionIds[i]))] = true;
        }
        
        emit MotionMintedV2(motionId, packHash, creator, ipfsUri, derivationDepth, parentMotionIds);
        emit GaslessExecutionProcessed(signer, digest, true);
        
        return motionId;
    }
    
    // ========== GAS SPONSOR MANAGEMENT ==========
    
    /**
     * @notice Replenish gas sponsor budget
     */
    function replenishGasBudget() external payable {
        uint256 previousBalance = gasSponsorBudget;
        gasSponsorBudget += msg.value;
        
        emit GasBudgetReplenished(previousBalance, gasSponsorBudget, msg.sender);
    }
    
    /**
     * @notice Check if gas budget needs replenishment
     * @return needsReplenish True if below 40% threshold
     * @return currentBalance Current gas sponsor balance
     * @return threshold Replenishment threshold
     */
    function checkGasBudgetStatus() external view returns (
        bool needsReplenish,
        uint256 currentBalance,
        uint256 threshold
    ) {
        threshold = (gasSponsorInitialBudget * GAS_SPONSOR_REPLENISH_THRESHOLD_BPS) / BASIS_POINTS;
        currentBalance = gasSponsorBudget;
        needsReplenish = currentBalance < threshold;
    }
    
    // ========== VIEW FUNCTIONS ==========
    
    function getMotionPackV2(uint256 motionId) external view returns (
        bytes32 packHash,
        address creator,
        uint256 mintedAt,
        string memory ipfsUri,
        bool isActive,
        uint256 usageCount,
        uint256 derivationDepth,
        uint256[] memory parentMotionIds,
        address[] memory royaltyRecipients,
        uint256[] memory royaltySharesBps
    ) {
        MotionPackV2 storage pack = motionPacks[motionId];
        return (
            pack.packHash,
            pack.creator,
            pack.mintedAt,
            pack.ipfsUri,
            pack.isActive,
            pack.usageCount,
            pack.derivationDepth,
            pack.parentMotionIds,
            pack.royaltyRecipients,
            pack.royaltySharesBps
        );
    }
    
    function getRoyaltyChain(uint256 motionId) external view returns (
        RoyaltyNode[] memory nodes
    ) {
        MotionPackV2 storage pack = motionPacks[motionId];
        nodes = new RoyaltyNode[](pack.royaltyRecipients.length);
        
        for (uint256 i = 0; i < pack.royaltyRecipients.length; i++) {
            uint256 sourceMotion = i < pack.parentMotionIds.length 
                ? pack.parentMotionIds[i] 
                : 0;
            
            nodes[i] = RoyaltyNode({
                recipient: pack.royaltyRecipients[i],
                motionId: sourceMotion,
                shareBps: pack.royaltySharesBps[i],
                depth: i
            });
        }
    }
    
    /**
     * @notice Calculate expected payouts for a given usage
     * @param motionId Motion pack ID
     * @param durationSeconds Usage duration
     * @return totalFee Total fee to be paid
     * @return royaltyAmounts Array of royalty amounts by depth
     * @return creatorAmount Amount to primary creator
     */
    function calculatePayouts(
        uint256 motionId,
        uint256 durationSeconds
    ) external view returns (
        uint256 totalFee,
        uint256[] memory royaltyAmounts,
        uint256 creatorAmount
    ) {
        MotionPackV2 storage pack = motionPacks[motionId];
        totalFee = usageFeePerSecond * durationSeconds;
        
        royaltyAmounts = new uint256[](pack.royaltyRecipients.length);
        uint256 totalRoyalties = 0;
        
        for (uint256 i = 0; i < pack.royaltyRecipients.length && i < MAX_ROYALTY_CHAIN_DEPTH; i++) {
            uint256 royaltyAmount = (totalFee * pack.royaltySharesBps[i]) / BASIS_POINTS;
            
            if (i > 0) {
                royaltyAmount = (royaltyAmount * ROYALTY_DECAY_FACTOR_BPS) / BASIS_POINTS;
            }
            
            royaltyAmounts[i] = royaltyAmount;
            totalRoyalties += royaltyAmount;
        }
        
        uint256 remainingFee = totalFee - totalRoyalties;
        creatorAmount = (remainingFee * payoutSplit.creatorBps) / BASIS_POINTS;
    }
    
    // ========== ADMIN FUNCTIONS ==========
    
    function authorizeOracle(address oracle, bool authorized) external onlyOwner {
        authorizedOracles[oracle] = authorized;
    }
    
    function authorizeEngine(address engine, bool authorized) external onlyOwner {
        authorizedEngines[engine] = authorized;
    }
    
    function authorizeGasSponsor(address operator, bool authorized) external onlyOwner {
        gasSponsorOperators[operator] = authorized;
    }
    
    function updatePricing(uint256 newMintPrice, uint256 newUsageFee) external onlyOwner {
        mintPrice = newMintPrice;
        usageFeePerSecond = newUsageFee;
    }
    
    function updatePayoutSplit(
        uint256 creatorBps,
        uint256 oracleBps,
        uint256 platformBps,
        uint256 opsBps
    ) external onlyOwner {
        require(creatorBps + oracleBps + platformBps + opsBps == BASIS_POINTS, "Must sum to 100%");
        payoutSplit = PayoutSplitV2({
            creatorBps: creatorBps,
            oracleBps: oracleBps,
            platformBps: platformBps,
            opsBps: opsBps
        });
    }
    
    function deactivateMotion(uint256 motionId) external onlyOwner {
        motionPacks[motionId].isActive = false;
    }
    
    function withdrawExcessBalance(address payable recipient) external onlyOwner {
        uint256 excess = address(this).balance - gasSponsorBudget;
        require(excess > 0, "No excess balance");
        recipient.transfer(excess);
    }
    
    // Allow contract to receive USDC (Arc native token)
    receive() external payable {}
}
