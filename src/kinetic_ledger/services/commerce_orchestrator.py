"""
Commerce Orchestrator - handles Circle payments and USDC settlement.

Enhanced with:
- x402 facilitator integration for gasless payments
- Royalty chain support for infinite-depth creator payouts
- Gasless executor for EIP-7702 sponsored transactions
- 40% auto-replenish gas sponsorship budget
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ..schemas.models import (
    UsageMeterEvent,
    Metering,
    X402,
    Settlement,
    PayoutItem,
    PayoutSplit,
    RoyaltyChain,
    RecursivePayoutResult,
    ROYALTY_DECAY_FACTOR,
    MAX_ROYALTY_CHAIN_DEPTH,
    GAS_SPONSOR_REPLENISH_THRESHOLD,
)
from ..utils.logging import setup_logging, set_correlation_id, get_logger
from ..utils.errors import CircleError, PolicyError, PaymentRequiredError, SettlementError
from ..utils.retry import retry_on_dependency_error

# Import new services (with optional fallback for backwards compatibility)
try:
    from .x402_facilitator import X402FacilitatorService, X402FacilitatorConfig
    from .gasless_executor import GaslessExecutor, GaslessExecutorConfig
    from .payment_automation import PaymentAutomationService, PaymentAutomationConfig
    HAS_NEW_SERVICES = True
except ImportError:
    HAS_NEW_SERVICES = False


logger = get_logger(__name__)


class CircleWalletConfig(BaseModel):
    """Circle wallet configuration."""
    api_key: str
    api_endpoint: str = "https://api.circle.com/v1"


class PaymentIntentRequest(BaseModel):
    """Request to create payment intent."""
    user_wallet: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    product: str
    unit: str = Field(pattern=r"^(seconds_generated|frames_generated|agent_steps)$")
    quantity: float = Field(ge=0)
    unit_price_usdc: str  # Decimal string
    attestation_pack_hash: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")


class CircleClient:
    """
    Circle API client wrapper.
    
    In production: integrate with Circle SDK for:
    - Wallet creation and management
    - Payment processing
    - USDC transfers via Gateway/CCTP
    
    For demo: simulate Circle API responses.
    """
    
    def __init__(self, config: CircleWalletConfig):
        self.config = config
    
    @retry_on_dependency_error(max_attempts=3)
    def create_wallet(self, user_id: str) -> Dict[str, Any]:
        """Create Circle wallet for user."""
        logger.info(f"Creating Circle wallet for user: {user_id}")
        
        # Simulate API call
        wallet_id = f"wallet_{uuid.uuid4().hex[:16]}"
        
        return {
            "wallet_id": wallet_id,
            "blockchain": "ETH",
            "address": f"0x{'0' * 39}{user_id[:1]}",
            "status": "active",
        }
    
    @retry_on_dependency_error(max_attempts=3)
    def create_payment_intent(
        self,
        wallet_id: str,
        amount_usdc: str,
        description: str,
    ) -> Dict[str, Any]:
        """Create payment intent."""
        logger.info(
            f"Creating payment intent: {amount_usdc} USDC",
            extra={"wallet_id": wallet_id},
        )
        
        # Simulate API call
        intent_id = f"pi_{uuid.uuid4().hex[:16]}"
        
        return {
            "intent_id": intent_id,
            "wallet_id": wallet_id,
            "amount": amount_usdc,
            "currency": "USDC",
            "status": "pending",
            "created_at": int(time.time()),
        }
    
    @retry_on_dependency_error(max_attempts=3)
    def execute_transfer(
        self,
        from_wallet_id: str,
        to_address: str,
        amount_usdc: str,
        chain: str = "ARC",
    ) -> Dict[str, Any]:
        """Execute USDC transfer on specified chain."""
        logger.info(
            f"Executing transfer: {amount_usdc} USDC to {to_address}",
            extra={"chain": chain},
        )
        
        # Simulate blockchain transaction
        tx_hash = f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"
        
        return {
            "tx_hash": tx_hash,
            "chain": chain,
            "from": from_wallet_id,
            "to": to_address,
            "amount": amount_usdc,
            "status": "confirmed",
            "block_number": 12345678,
        }


class X402Verifier:
    """
    x402 payment proof verifier.
    
    In production: integrate with x402 facilitator to verify payment proofs.
    """
    
    def __init__(self, facilitator_url: str = "https://x402.example.com"):
        self.facilitator_url = facilitator_url
    
    @retry_on_dependency_error(max_attempts=3)
    def verify_payment_proof(
        self,
        proof: str,
        expected_amount: str,
    ) -> Dict[str, Any]:
        """
        Verify x402 payment proof.
        
        Args:
            proof: Payment proof string
            expected_amount: Expected USDC amount
        
        Returns:
            Verification result
        """
        logger.info(f"Verifying x402 payment proof")
        
        # Simulate verification
        receipt_id = f"x402_{uuid.uuid4().hex[:16]}"
        
        return {
            "verified": True,
            "receipt_id": receipt_id,
            "amount": expected_amount,
            "timestamp": int(time.time()),
        }


class CommerceOrchestrator:
    """
    Commerce Orchestrator Service.
    
    Responsibilities:
    1. Create and manage Circle wallets
    2. Process payment intents for usage-based billing
    3. Verify x402 payment proofs
    4. Execute USDC settlement on Arc
    5. Route payouts to creators/oracle/platform/ops
    6. Support infinite-depth royalty chains with recursive payouts
    7. Gasless transaction execution via EIP-7702
    
    Enhanced Features (when HAS_NEW_SERVICES=True):
    - X402FacilitatorService for real x402 verification/settlement
    - GaslessExecutor for EIP-7702 sponsored transactions
    - PaymentAutomationService for auto-payouts after on-chain confirmation
    - 40% gas sponsor replenish threshold
    """
    
    def __init__(
        self,
        circle_config: CircleWalletConfig,
        x402_facilitator_url: str = "https://x402-facilitator.thirdweb.com",
        usdc_token_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        payout_split: Optional[PayoutSplit] = None,
        # New service instances (optional for backwards compatibility)
        x402_facilitator: Optional[Any] = None,
        gasless_executor: Optional[Any] = None,
        payment_automation: Optional[Any] = None,
    ):
        # Legacy stubbed clients (fallback)
        self.circle_client = CircleClient(circle_config)
        self.x402_verifier = X402Verifier(x402_facilitator_url)
        self.usdc_token_address = usdc_token_address
        self.payout_split = payout_split or PayoutSplit(
            creator=0.70,
            oracle=0.10,
            platform=0.15,
            ops=0.05,
        )
        
        # New services integration (enhanced mode)
        self._x402_facilitator = x402_facilitator
        self._gasless_executor = gasless_executor
        self._payment_automation = payment_automation
        
        # Try to auto-initialize new services if available
        if HAS_NEW_SERVICES:
            self._init_new_services()
        
        setup_logging("commerce-orchestrator")
    
    def _init_new_services(self) -> None:
        """Initialize new payment services if available and not already set."""
        try:
            if self._x402_facilitator is None:
                config = X402FacilitatorConfig()
                self._x402_facilitator = X402FacilitatorService(config)
                logger.info("[INIT] X402FacilitatorService initialized")
            
            if self._gasless_executor is None:
                config = GaslessExecutorConfig()
                self._gasless_executor = GaslessExecutor(config)
                logger.info("[INIT] GaslessExecutor initialized")
            
            if self._payment_automation is None:
                config = PaymentAutomationConfig()
                self._payment_automation = PaymentAutomationService(config)
                logger.info("[INIT] PaymentAutomationService initialized")
        except Exception as e:
            logger.warning(f"[INIT] Failed to initialize new services: {e}")
    
    @property
    def has_enhanced_services(self) -> bool:
        """Check if enhanced services are available."""
        return (
            HAS_NEW_SERVICES
            and self._x402_facilitator is not None
            and self._gasless_executor is not None
            and self._payment_automation is not None
        )
    
    def _calculate_payouts(
        self,
        total_usdc: float,
        creator_address: str,
        oracle_address: str,
        platform_address: str,
        ops_address: str,
    ) -> List[PayoutItem]:
        """Calculate payout distribution."""
        return [
            PayoutItem(
                to=creator_address,
                amount_usdc=f"{total_usdc * self.payout_split.creator:.6f}",
                label="creator",
            ),
            PayoutItem(
                to=oracle_address,
                amount_usdc=f"{total_usdc * self.payout_split.oracle:.6f}",
                label="oracle",
            ),
            PayoutItem(
                to=platform_address,
                amount_usdc=f"{total_usdc * self.payout_split.platform:.6f}",
                label="platform",
            ),
            PayoutItem(
                to=ops_address,
                amount_usdc=f"{total_usdc * self.payout_split.ops:.6f}",
                label="ops",
            ),
        ]
    
    def meter_usage(
        self,
        request: PaymentIntentRequest,
        payment_proof: str,
        creator_address: str,
        oracle_address: str = "0x0000000000000000000000000000000000000001",
        platform_address: str = "0x0000000000000000000000000000000000000002",
        ops_address: str = "0x0000000000000000000000000000000000000003",
        correlation_id: Optional[str] = None,
        royalty_chain: Optional[RoyaltyChain] = None,
        use_gasless: bool = False,
    ) -> UsageMeterEvent:
        """
        Meter usage and execute USDC settlement.
        
        Enhanced with:
        - Real x402 verification via X402FacilitatorService
        - Recursive royalty payout for derivative content
        - Gasless execution via EIP-7702
        
        Args:
            request: Payment intent request
            payment_proof: x402 payment proof
            creator_address: Creator wallet address
            oracle_address: Oracle wallet address
            platform_address: Platform wallet address
            ops_address: Ops wallet address
            correlation_id: Correlation ID
            royalty_chain: Optional royalty chain for derivative content
            use_gasless: Whether to use gasless execution
        
        Returns:
            UsageMeterEvent with settlement details
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        usage_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        logger.info(
            f"[ENTRY] meter_usage: {request.quantity} {request.unit}",
            extra={"usage_id": usage_id, "product": request.product},
        )
        
        # Calculate total amount
        unit_price = float(request.unit_price_usdc)
        total_usdc = unit_price * request.quantity
        total_usdc_str = f"{total_usdc:.6f}"
        
        # Verify x402 payment proof (use enhanced or legacy)
        verification = self._verify_payment(
            payment_proof=payment_proof,
            expected_amount=total_usdc_str,
        )
        
        if not verification.get("verified", False):
            raise PaymentRequiredError(
                f"Payment proof verification failed",
                details={"expected": total_usdc_str},
            )
        
        # Execute settlement (use gasless if enabled and available)
        if use_gasless and self.has_enhanced_services:
            settlement_result = self._execute_gasless_settlement(
                total_usdc=total_usdc,
                creator_address=creator_address,
                royalty_chain=royalty_chain,
            )
            tx_hash = settlement_result.get("tx_hash", "")
        else:
            # Legacy: simulate settlement
            tx_hash = f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"
        
        logger.info(
            f"Executing USDC settlement on Arc",
            extra={"tx_hash": tx_hash, "amount": total_usdc_str},
        )
        
        # Calculate payouts (with royalty chain if provided)
        if royalty_chain and self.has_enhanced_services:
            payouts = self._calculate_payouts_with_royalties(
                total_usdc=total_usdc,
                creator_address=creator_address,
                oracle_address=oracle_address,
                platform_address=platform_address,
                ops_address=ops_address,
                royalty_chain=royalty_chain,
            )
        else:
            payouts = self._calculate_payouts(
                total_usdc=total_usdc,
                creator_address=creator_address,
                oracle_address=oracle_address,
                platform_address=platform_address,
                ops_address=ops_address,
            )
        
        # Trigger auto-payout after settlement (enhanced mode)
        if self.has_enhanced_services and self._payment_automation:
            try:
                self._trigger_auto_payout(
                    tx_hash=tx_hash,
                    total_usdc=total_usdc,
                    creator_address=creator_address,
                    royalty_chain=royalty_chain,
                )
            except Exception as e:
                logger.warning(f"Auto-payout trigger failed: {e}")
        
        # Create usage meter event
        event = UsageMeterEvent(
            usage_id=usage_id,
            created_at=timestamp,
            user_wallet=request.user_wallet,
            attestation_pack_hash=request.attestation_pack_hash,
            product=request.product,
            metering=Metering(
                unit=request.unit,
                quantity=request.quantity,
                unit_price_usdc=request.unit_price_usdc,
                total_usdc=total_usdc_str,
            ),
            x402=X402(
                payment_proof=payment_proof,
                facilitator_receipt_id=verification.get("receipt_id", ""),
                verified=True,
            ),
            settlement=Settlement(
                chain="ARC",
                token=self.usdc_token_address,
                tx_hash=tx_hash,
            ),
            payout_split=payouts,
        )
        
        logger.info(
            f"[EXIT] meter_usage: success",
            extra={
                "usage_id": usage_id,
                "total_usdc": total_usdc_str,
                "tx_hash": tx_hash,
                "payouts": len(payouts),
            },
        )
        
        return event
    
    def _verify_payment(
        self,
        payment_proof: str,
        expected_amount: str,
    ) -> Dict[str, Any]:
        """
        Verify payment proof using enhanced or legacy verifier.
        
        Args:
            payment_proof: x402 payment proof or header
            expected_amount: Expected USDC amount
        
        Returns:
            Verification result dict
        """
        if self.has_enhanced_services and self._x402_facilitator:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    self._x402_facilitator.verify_payment(
                        x402_header=payment_proof,
                        expected_amount=expected_amount,
                    )
                )
                return {
                    "verified": result.valid,
                    "receipt_id": result.receipt_id,
                    "amount": expected_amount,
                    "timestamp": int(time.time()),
                }
            except Exception as e:
                logger.warning(f"Enhanced verification failed, falling back: {e}")
        
        # Fallback to legacy verifier
        return self.x402_verifier.verify_payment_proof(
            proof=payment_proof,
            expected_amount=expected_amount,
        )
    
    def _execute_gasless_settlement(
        self,
        total_usdc: float,
        creator_address: str,
        royalty_chain: Optional[RoyaltyChain] = None,
    ) -> Dict[str, Any]:
        """
        Execute gasless settlement via EIP-7702.
        
        Args:
            total_usdc: Total USDC amount
            creator_address: Primary creator address
            royalty_chain: Optional royalty chain
        
        Returns:
            Settlement result with tx_hash
        """
        if not self._gasless_executor:
            raise SettlementError("Gasless executor not available")
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            if royalty_chain:
                # Use batch recursive transfer for royalty chains
                result = loop.run_until_complete(
                    self._gasless_executor.sponsor_royalty_payout(
                        royalty_chain=royalty_chain,
                        base_amount=total_usdc,
                    )
                )
                return {
                    "tx_hash": result.receipts[0].tx_hash if result.receipts else "",
                    "gasless": True,
                    "total_payouts": result.total_paid,
                }
            else:
                # Single transfer
                result = loop.run_until_complete(
                    self._gasless_executor.send_gasless_tx(
                        to_address=creator_address,
                        amount_usdc=total_usdc,
                    )
                )
                return {
                    "tx_hash": result.tx_hash,
                    "gasless": True,
                }
        except Exception as e:
            logger.error(f"Gasless settlement failed: {e}")
            raise SettlementError(f"Gasless settlement failed: {e}")
    
    def _calculate_payouts_with_royalties(
        self,
        total_usdc: float,
        creator_address: str,
        oracle_address: str,
        platform_address: str,
        ops_address: str,
        royalty_chain: RoyaltyChain,
    ) -> List[PayoutItem]:
        """
        Calculate payouts including royalty chain distribution.
        
        Uses ROYALTY_DECAY_FACTOR (0.5) for each level in the chain.
        
        Args:
            total_usdc: Total payment amount
            creator_address: Primary creator
            oracle_address: Oracle address
            platform_address: Platform address
            ops_address: Ops address
            royalty_chain: Royalty chain with parent obligations
        
        Returns:
            List of PayoutItems including royalty recipients
        """
        payouts: List[PayoutItem] = []
        
        # Calculate base split after royalties
        total_royalty = sum(
            node.royalty_percentage for node in royalty_chain.nodes
        )
        creator_share = max(0, self.payout_split.creator - total_royalty)
        
        # Primary creator payout (reduced by royalty obligations)
        if creator_share > 0:
            payouts.append(PayoutItem(
                to=creator_address,
                amount_usdc=f"{total_usdc * creator_share:.6f}",
                label="creator",
            ))
        
        # Royalty payouts with decay
        for i, node in enumerate(royalty_chain.nodes[:MAX_ROYALTY_CHAIN_DEPTH]):
            decay_factor = ROYALTY_DECAY_FACTOR ** i
            royalty_amount = total_usdc * node.royalty_percentage * decay_factor
            
            if royalty_amount >= 0.01:  # Minimum $0.01 payout
                payouts.append(PayoutItem(
                    to=node.wallet_address,
                    amount_usdc=f"{royalty_amount:.6f}",
                    label=f"royalty_depth_{i}",
                ))
        
        # Standard splits for oracle, platform, ops
        payouts.extend([
            PayoutItem(
                to=oracle_address,
                amount_usdc=f"{total_usdc * self.payout_split.oracle:.6f}",
                label="oracle",
            ),
            PayoutItem(
                to=platform_address,
                amount_usdc=f"{total_usdc * self.payout_split.platform:.6f}",
                label="platform",
            ),
            PayoutItem(
                to=ops_address,
                amount_usdc=f"{total_usdc * self.payout_split.ops:.6f}",
                label="ops",
            ),
        ])
        
        return payouts
    
    def _trigger_auto_payout(
        self,
        tx_hash: str,
        total_usdc: float,
        creator_address: str,
        royalty_chain: Optional[RoyaltyChain] = None,
    ) -> None:
        """
        Trigger automatic payout after on-chain settlement confirmation.
        
        Args:
            tx_hash: Settlement transaction hash
            total_usdc: Total USDC amount
            creator_address: Primary creator address
            royalty_chain: Optional royalty chain
        """
        if not self._payment_automation:
            return
        
        try:
            import asyncio
            from ..schemas.models import PaymentTriggerEvent
            
            loop = asyncio.get_event_loop()
            
            trigger_event = PaymentTriggerEvent(
                motion_id="auto_" + tx_hash[:16],
                arc_tx_hash=tx_hash,
                total_usdc=total_usdc,
                creator_address=creator_address,
                royalty_chain=royalty_chain,
            )
            
            loop.run_until_complete(
                self._payment_automation.trigger_creator_payout(trigger_event)
            )
            
            logger.info(f"Auto-payout triggered for tx: {tx_hash[:16]}")
        except Exception as e:
            logger.warning(f"Auto-payout failed: {e}")
    
    def create_subscription(
        self,
        user_wallet: str,
        plan: str,
        monthly_price_usdc: str,
    ) -> Dict[str, Any]:
        """
        Create subscription for creator content.
        
        Args:
            user_wallet: User wallet address
            plan: Subscription plan name
            monthly_price_usdc: Monthly price in USDC
        
        Returns:
            Subscription details
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:16]}"
        
        logger.info(
            f"Creating subscription: {plan}",
            extra={
                "subscription_id": subscription_id,
                "user_wallet": user_wallet,
                "price": monthly_price_usdc,
            },
        )
        
        # In production: create recurring payment intent
        
        return {
            "subscription_id": subscription_id,
            "user_wallet": user_wallet,
            "plan": plan,
            "monthly_price": monthly_price_usdc,
            "status": "active",
            "next_billing_date": int(time.time()) + (30 * 24 * 3600),
        }
    
    async def settle_with_royalty_chain(
        self,
        x402_header: str,
        expected_amount: float,
        creator_address: str,
        royalty_chain: RoyaltyChain,
        use_gasless: bool = True,
    ) -> RecursivePayoutResult:
        """
        Full x402 settlement with infinite-depth royalty chain support.
        
        This is the primary entry point for derivative motion commerce:
        1. Verify x402 payment proof
        2. Build recursive payout distribution
        3. Execute gasless settlement on Arc
        4. Auto-trigger creator payouts after confirmation
        
        Args:
            x402_header: x402 payment header from HTTP request
            expected_amount: Expected USDC amount
            creator_address: Primary creator wallet address
            royalty_chain: RoyaltyChain with all ancestors
            use_gasless: Whether to use EIP-7702 gasless execution
        
        Returns:
            RecursivePayoutResult with all payout receipts
        
        Raises:
            PaymentRequiredError: If x402 verification fails
            SettlementError: If settlement execution fails
        """
        if not self.has_enhanced_services:
            raise SettlementError("Enhanced services required for royalty chain settlement")
        
        logger.info(
            f"[ENTRY] settle_with_royalty_chain: {expected_amount} USDC, "
            f"chain_depth={len(royalty_chain.nodes)}"
        )
        
        # Step 1: Verify x402 payment
        verification = await self._x402_facilitator.verify_payment(
            x402_header=x402_header,
            expected_amount=f"{expected_amount:.6f}",
        )
        
        if not verification.valid:
            raise PaymentRequiredError(
                "x402 payment verification failed",
                details={"expected": expected_amount, "reason": verification.error_message},
            )
        
        # Step 2: Settle and get receipt
        settlement = await self._x402_facilitator.settle_payment(
            x402_header=x402_header,
        )
        
        if not settlement.success:
            raise SettlementError(f"x402 settlement failed: {settlement.error_message}")
        
        # Step 3: Execute royalty payouts
        if use_gasless and self._gasless_executor:
            # Gasless batch payout
            result = await self._gasless_executor.sponsor_royalty_payout(
                royalty_chain=royalty_chain,
                base_amount=expected_amount,
            )
        elif self._payment_automation:
            # Standard payout via Circle
            payouts = await self._payment_automation.calculate_recursive_payouts(
                royalty_chain=royalty_chain,
                base_amount=expected_amount,
            )
            result = await self._payment_automation.process_royalty_payments(payouts)
        else:
            raise SettlementError("No payout executor available")
        
        logger.info(
            f"[EXIT] settle_with_royalty_chain: "
            f"total_paid={result.total_paid}, levels={result.levels_processed}"
        )
        
        return result
    
    async def check_and_replenish_gas_budget(self) -> Dict[str, Any]:
        """
        Check gas sponsor budget and replenish if below threshold.
        
        Uses GAS_SPONSOR_REPLENISH_THRESHOLD (40%) to determine when to replenish.
        
        Returns:
            Dict with current_balance, threshold_met, replenished status
        """
        if not self._gasless_executor:
            return {"error": "Gasless executor not available"}
        
        balance = await self._gasless_executor.get_gas_budget_balance()
        threshold = self._gasless_executor.config.initial_budget * GAS_SPONSOR_REPLENISH_THRESHOLD
        
        result = {
            "current_balance": balance,
            "threshold": threshold,
            "threshold_met": balance >= threshold,
            "replenished": False,
        }
        
        if balance < threshold:
            logger.warning(
                f"Gas budget below threshold: {balance:.4f} < {threshold:.4f}"
            )
            await self._gasless_executor.replenish_gas_budget()
            result["replenished"] = True
            result["new_balance"] = await self._gasless_executor.get_gas_budget_balance()
        
        return result
