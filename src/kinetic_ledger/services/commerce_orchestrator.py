"""
Commerce Orchestrator - handles Circle payments and USDC settlement.
Integrates with Circle's Wallets API for motion commerce payments.
"""
import logging
import time
import uuid
import os
import requests
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
    api_endpoint: str = "https://api.circle.com/v1"  # Production endpoint
    
    @classmethod
    def from_env(cls):
        """Load from environment variables."""
        api_key = os.getenv("CIRCLE_API_KEY")
        if not api_key:
            raise CircleError("CIRCLE_API_KEY not set - using fallback mode")
        return cls(api_key=api_key)


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
    Circle API client for Wallets and payment processing.
    
    Integrates with Circle's RESTful API for:
    - Wallet creation and management
    - Payment processing
    - USDC transfers
    
    Uses Bearer token authentication with Circle API keys.
    """
    
    def __init__(self, config: CircleWalletConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Circle API."""
        url = f"{self.config.api_endpoint}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise CircleError("Circle API authentication failed - check API key")
            raise CircleError(f"Circle API error: {e}")
        except Exception as e:
            raise CircleError(f"Circle API request failed: {e}")
    
    @retry_on_dependency_error(max_attempts=3)
    def get_wallets(self) -> Dict[str, Any]:
        """Retrieve wallets to verify API connectivity."""
        logger.info("Fetching Circle wallets")
        return self._make_request("GET", "/w3s/wallets")
    
    @retry_on_dependency_error(max_attempts=3)
    def create_wallet(self, user_id: str, blockchain: str = "ETH-SEPOLIA") -> Dict[str, Any]:
        """
        Create Circle Developer-Controlled Wallet for motion commerce.
        
        Uses Circle's Programmable Wallets API v1/w3s/wallets endpoint.
        Developer-Controlled Wallets are ideal for motion commerce as they:
        - Require only API key (no client key needed)
        - Support USDC transfers on Arc Network
        - Enable automated royalty distribution
        
        Args:
            user_id: User identifier for wallet metadata
            blockchain: Blockchain network (default: ETH-SEPOLIA for Arc testnet)
        
        Returns:
            Wallet creation response with wallet_id and address
        """
        logger.info(f"Creating Circle wallet for user: {user_id}")
        
        payload = {
            "idempotencyKey": f"wallet-{user_id}-{uuid.uuid4().hex[:8]}",
            "accountType": "SCA",  # Smart Contract Account
            "blockchain": blockchain,
            "metadata": [
                {"key": "user_id", "value": user_id},
                {"key": "purpose", "value": "motion_commerce"}
            ]
        }
        
        response = self._make_request("POST", "/w3s/wallets", json=payload)
        
        # Circle response format:
        # {
        #   "data": {
        #     "walletId": "...",
        #     "address": "0x...",
        #     "blockchain": "ETH-SEPOLIA",
        #     "accountType": "SCA",
        #     "state": "LIVE"
        #   }
        # }
        
        wallet_data = response.get("data", {})
        return {
            "wallet_id": wallet_data.get("walletId"),
            "blockchain": wallet_data.get("blockchain"),
            "address": wallet_data.get("address"),
            "status": wallet_data.get("state", "").lower(),
        }
    
    @retry_on_dependency_error(max_attempts=3)
    def create_payment_intent(
        self,
        wallet_id: str,
        amount_usdc: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Create Circle payment intent for motion usage billing.
        
        Note: Circle's Payments API may use different endpoint structure.
        This is a conceptual implementation - actual endpoint TBD from Circle docs.
        For motion commerce, may use direct transfers instead of payment intents.
        """
        logger.info(
            f"Creating payment intent: {amount_usdc} USDC",
            extra={"wallet_id": wallet_id},
        )
        
        # Note: Actual Circle API may not have a payment intents endpoint
        # For developer-controlled wallets, we typically execute direct transfers
        # Keeping stub for now until Circle API documentation confirms endpoint
        
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
        token_id: str = "36b1737c-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # USDC token ID
        blockchain: str = "ETH-SEPOLIA",
    ) -> Dict[str, Any]:
        """
        Execute USDC transfer via Circle Programmable Wallets API.
        
        Uses /w3s/developer/transactions/transfer endpoint for outbound transfers.
        
        Args:
            from_wallet_id: Source wallet ID (must be developer-controlled)
            to_address: Destination blockchain address
            amount_usdc: Amount in USDC (e.g., "10.50")
            token_id: Circle token ID for USDC on target blockchain
            blockchain: Target blockchain (ETH-SEPOLIA for Arc testnet)
        
        Returns:
            Transfer response with transaction ID and status
        """
        logger.info(
            f"Executing transfer: {amount_usdc} USDC to {to_address}",
            extra={"blockchain": blockchain},
        )
        
        # Convert USDC amount to smallest unit (6 decimals)
        # Example: "10.50" -> "10500000"
        amount_smallest_unit = str(int(float(amount_usdc) * 1_000_000))
        
        payload = {
            "idempotencyKey": f"transfer-{uuid.uuid4().hex}",
            "walletId": from_wallet_id,
            "tokenId": token_id,
            "destinationAddress": to_address,
            "amounts": [amount_smallest_unit],
            "fee": {
                "type": "level",
                "config": {
                    "feeLevel": "MEDIUM"
                }
            }
        }
        
        response = self._make_request(
            "POST",
            "/w3s/developer/transactions/transfer",
            json=payload
        )
        
        # Circle response format:
        # {
        #   "data": {
        #     "id": "transaction-id",
        #     "state": "INITIATED",
        #     "walletId": "...",
        #     "blockchain": "ETH-SEPOLIA",
        #     "txHash": "0x..." (after confirmation)
        #   }
        # }
        
        transfer_data = response.get("data", {})
        return {
            "tx_id": transfer_data.get("id"),
            "tx_hash": transfer_data.get("txHash"),
            "blockchain": transfer_data.get("blockchain"),
            "from": from_wallet_id,
            "to": to_address,
            "amount": amount_usdc,
            "status": transfer_data.get("state", "").lower(),
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
        circle_config: Optional[CircleWalletConfig] = None,
        x402_facilitator_url: str = "https://x402.example.com",
        usdc_token_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        payout_split: Optional[PayoutSplit] = None,
    ):
        # Initialize Circle client - use env config if not provided
        if circle_config is None:
            try:
                circle_config = CircleWalletConfig.from_env()
            except CircleError:
                # Fallback mode without Circle integration
                logger.warning("Circle API not configured - running in fallback mode")
                circle_config = CircleWalletConfig(api_key="fallback")
        
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
    
    def create_user_wallet(self, user_id: str, blockchain: str = "ETH-SEPOLIA") -> Dict[str, Any]:
        """
        Create Circle wallet for user.
        
        Args:
            user_id: User identifier
            blockchain: Target blockchain network
        
        Returns:
            Wallet creation result
        """
        logger.info(f"Creating wallet for user: {user_id}")
        return self.circle_client.create_wallet(user_id, blockchain)
    
    def process_payment(
        self,
        wallet_id: str,
        amount_usdc: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Process payment intent.
        
        Args:
            wallet_id: Source wallet ID
            amount_usdc: Payment amount in USDC
            description: Payment description
        
        Returns:
            Payment intent result
        """
        logger.info(f"Processing payment: {amount_usdc} USDC")
        return self.circle_client.create_payment_intent(
            wallet_id=wallet_id,
            amount_usdc=amount_usdc,
            description=description,
        )
    
    def distribute_royalty(
        self,
        from_wallet_id: str,
        creator_address: str,
        amount_usdc: str,
        blockchain: str = "ETH-SEPOLIA",
    ) -> Dict[str, Any]:
        """
        Distribute royalty payment to creator.
        
        Args:
            from_wallet_id: Source wallet ID (platform wallet)
            creator_address: Creator's blockchain address
            amount_usdc: Royalty amount in USDC
            blockchain: Target blockchain
        
        Returns:
            Transfer result
        """
        logger.info(f"Distributing royalty: {amount_usdc} USDC to {creator_address}")
        return self.circle_client.execute_transfer(
            from_wallet_id=from_wallet_id,
            to_address=creator_address,
            amount_usdc=amount_usdc,
            blockchain=blockchain,
        )
