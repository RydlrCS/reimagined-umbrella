"""
Commerce Orchestrator - handles Circle payments and USDC settlement.
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
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import CircleError, PolicyError, PaymentRequiredError, SettlementError
from ..utils.retry import retry_on_dependency_error


logger = logging.getLogger(__name__)


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
    """
    
    def __init__(
        self,
        circle_config: CircleWalletConfig,
        x402_facilitator_url: str = "https://x402.example.com",
        usdc_token_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        payout_split: Optional[PayoutSplit] = None,
    ):
        self.circle_client = CircleClient(circle_config)
        self.x402_verifier = X402Verifier(x402_facilitator_url)
        self.usdc_token_address = usdc_token_address
        self.payout_split = payout_split or PayoutSplit(
            creator=0.70,
            oracle=0.10,
            platform=0.15,
            ops=0.05,
        )
        setup_logging("commerce-orchestrator")
    
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
    ) -> UsageMeterEvent:
        """
        Meter usage and execute USDC settlement.
        
        Args:
            request: Payment intent request
            payment_proof: x402 payment proof
            creator_address: Creator wallet address
            oracle_address: Oracle wallet address
            platform_address: Platform wallet address
            ops_address: Ops wallet address
            correlation_id: Correlation ID
        
        Returns:
            UsageMeterEvent with settlement details
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        usage_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        logger.info(
            f"Metering usage: {request.quantity} {request.unit}",
            extra={"usage_id": usage_id, "product": request.product},
        )
        
        # Calculate total amount
        unit_price = float(request.unit_price_usdc)
        total_usdc = unit_price * request.quantity
        total_usdc_str = f"{total_usdc:.6f}"
        
        # Verify x402 payment proof
        verification = self.x402_verifier.verify_payment_proof(
            proof=payment_proof,
            expected_amount=total_usdc_str,
        )
        
        if not verification["verified"]:
            raise PaymentRequiredError(
                f"Payment proof verification failed",
                details={"expected": total_usdc_str},
            )
        
        # Execute settlement on Arc
        # In production: call Arc RPC and submit transaction
        tx_hash = f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"
        
        logger.info(
            f"Executing USDC settlement on Arc",
            extra={"tx_hash": tx_hash, "amount": total_usdc_str},
        )
        
        # Calculate payouts
        payouts = self._calculate_payouts(
            total_usdc=total_usdc,
            creator_address=creator_address,
            oracle_address=oracle_address,
            platform_address=platform_address,
            ops_address=ops_address,
        )
        
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
                facilitator_receipt_id=verification["receipt_id"],
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
            f"Usage metered successfully",
            extra={
                "usage_id": usage_id,
                "total_usdc": total_usdc_str,
                "tx_hash": tx_hash,
                "payouts": len(payouts),
            },
        )
        
        return event
    
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
