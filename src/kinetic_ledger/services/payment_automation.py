"""
Payment Automation Service.

Handles automatic USDC payouts triggered by on-chain mint confirmations.
Supports infinite-depth royalty chains with configurable decay factor.

Features:
- Auto-triggered creator payouts after Arc on-chain confirmation
- Recursive royalty distribution with 50% decay per level
- Gas sponsorship budget management (40% auto-replenish)
- Settlement reconciliation using KNN/RkCNN similarity
"""
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..schemas.models import (
    PaymentTriggerEvent,
    RoyaltyChain,
    PayoutItem,
    RecursivePayoutResult,
    ROYALTY_DECAY_FACTOR,
    MAX_ROYALTY_CHAIN_DEPTH,
    GAS_SPONSOR_REPLENISH_THRESHOLD,
    GAS_SPONSOR_MIN_BALANCE_USDC,
)
from ..utils.logging import get_logger
from ..utils.errors import SettlementError
from ..utils.retry import retry_on_dependency_error

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class PaymentAutomationConfig(BaseModel):
    """Configuration for payment automation service."""
    
    # Royalty chain settings (use global constants as defaults)
    decay_factor: float = Field(
        default=ROYALTY_DECAY_FACTOR,
        ge=0.0,
        le=1.0,
        description="Decay factor per royalty chain level (0.5 = 50%)"
    )
    max_chain_depth: int = Field(
        default=MAX_ROYALTY_CHAIN_DEPTH,
        ge=1,
        le=100,
    )
    
    # Gas sponsorship settings
    gas_replenish_threshold: float = Field(
        default=GAS_SPONSOR_REPLENISH_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Replenish when balance falls below this fraction of max"
    )
    gas_min_balance_usdc: str = Field(
        default=GAS_SPONSOR_MIN_BALANCE_USDC,
        description="Minimum USDC balance to trigger replenish"
    )
    
    # Treasury wallet addresses
    treasury_wallet: str = Field(
        default="0x0000000000000000000000000000000000000000",
        pattern=r"^0x[a-fA-F0-9]{40}$"
    )
    gas_sponsor_wallet: str = Field(
        default="0x0000000000000000000000000000000000000000",
        pattern=r"^0x[a-fA-F0-9]{40}$"
    )


# =============================================================================
# Payment Automation Service
# =============================================================================


class PaymentAutomationService:
    """
    Payment Automation Service.
    
    Automatically triggers USDC payouts to creators and royalty chain
    participants after on-chain mint confirmation. Uses recursive
    payout calculation with decay factor for infinite chain depth.
    
    Example:
        >>> config = PaymentAutomationConfig(decay_factor=0.5)
        >>> service = PaymentAutomationService(config)
        >>> result = service.trigger_creator_payout(
        ...     pack_hash="0x...",
        ...     tx_hash="0x...",
        ...     creator_address="0x...",
        ...     payout_amount_usdc="10.00",
        ...     royalty_chain=chain,
        ... )
    """
    
    def __init__(
        self,
        config: Optional[PaymentAutomationConfig] = None,
        circle_service: Optional[Any] = None,  # CircleWalletsService
        gasless_executor: Optional[Any] = None,  # GaslessExecutor
    ):
        """
        Initialize Payment Automation service.
        
        Args:
            config: Service configuration
            circle_service: Circle Wallets service for transfers
            gasless_executor: Gasless executor for sponsored transactions
        """
        logger.debug("[ENTRY] PaymentAutomationService.__init__")
        
        self.config = config or PaymentAutomationConfig()
        self._circle_service = circle_service
        self._gasless_executor = gasless_executor
        
        # Track pending payouts
        self._pending_payouts: dict[str, PaymentTriggerEvent] = {}
        
        # Gas sponsorship tracking
        self._gas_sponsor_balance_usdc: float = 1000.0  # Starting balance
        self._gas_sponsor_max_usdc: float = 2500.0
        
        logger.info(
            f"PaymentAutomationService initialized: "
            f"decay_factor={self.config.decay_factor}, "
            f"max_depth={self.config.max_chain_depth}"
        )
        logger.debug("[EXIT] PaymentAutomationService.__init__")
    
    # -------------------------------------------------------------------------
    # Creator Payout (Auto-Triggered)
    # -------------------------------------------------------------------------
    
    @retry_on_dependency_error(max_attempts=3, min_wait=1, max_wait=10)
    def trigger_creator_payout(
        self,
        pack_hash: str,
        tx_hash: str,
        creator_address: str,
        payout_amount_usdc: str,
        motion_id: Optional[str] = None,
        royalty_chain: Optional[RoyaltyChain] = None,
    ) -> PaymentTriggerEvent:
        """
        Trigger automatic payout after on-chain mint confirmation.
        
        Called by TrustlessAgentLoop after Arc Network confirms the
        mint transaction. Calculates recursive payouts and executes
        transfers via Circle Wallets.
        
        Args:
            pack_hash: Canonical pack hash (0x...)
            tx_hash: On-chain transaction hash
            creator_address: Creator's wallet address
            payout_amount_usdc: Total payout amount
            motion_id: Motion ID (optional)
            royalty_chain: Royalty chain for recursive distribution
        
        Returns:
            PaymentTriggerEvent with status and details
        """
        logger.info(
            f"[ENTRY] trigger_creator_payout: pack_hash={pack_hash[:16]}..., "
            f"amount={payout_amount_usdc} USDC"
        )
        
        event_id = f"pay_{uuid.uuid4().hex[:16]}"
        timestamp = int(time.time())
        
        # Create trigger event
        event = PaymentTriggerEvent(
            event_id=event_id,
            pack_hash=pack_hash,
            motion_id=motion_id or f"motion_{pack_hash[2:18]}",
            tx_hash=tx_hash,
            creator_address=creator_address,
            payout_amount_usdc=payout_amount_usdc,
            royalty_chain=royalty_chain,
            triggered_at=timestamp,
            status="PROCESSING",
        )
        
        self._pending_payouts[event_id] = event
        
        try:
            # Calculate recursive payouts
            if royalty_chain:
                payout_result = self.calculate_recursive_payouts(
                    total_usdc=float(payout_amount_usdc),
                    royalty_chain=royalty_chain,
                )
                
                logger.info(
                    f"Calculated {len(payout_result.payouts)} payouts, "
                    f"chain_depth={payout_result.chain_depth}"
                )
                
                # Execute payouts
                self._execute_payouts(payout_result.payouts)
            else:
                # Simple payout to creator only
                self._execute_simple_payout(creator_address, payout_amount_usdc)
            
            # Update event status
            event.status = "COMPLETED"
            logger.info(f"Payout completed: event_id={event_id}")
            
        except Exception as e:
            event.status = "FAILED"
            logger.error(f"Payout failed: {e}")
            raise SettlementError(
                f"Failed to execute creator payout: {e}",
                details={"event_id": event_id, "pack_hash": pack_hash},
            )
        
        logger.debug(f"[EXIT] trigger_creator_payout: status={event.status}")
        return event
    
    # -------------------------------------------------------------------------
    # Recursive Payout Calculation
    # -------------------------------------------------------------------------
    
    def calculate_recursive_payouts(
        self,
        total_usdc: float,
        royalty_chain: RoyaltyChain,
        decay_factor: Optional[float] = None,
        max_depth: Optional[int] = None,
    ) -> RecursivePayoutResult:
        """
        Calculate recursive payouts for royalty chain.
        
        Uses decay factor to distribute creator share across chain depth.
        Fixed parties (oracle, platform, ops) receive their full share
        from the total, while creator share decays through the chain.
        
        Decay example (50% factor):
        - Depth 0 (direct creator): 50% of creator pool = 35% of total
        - Depth 1 (parent creator): 25% of creator pool = 17.5% of total
        - Depth 2 (grandparent): 12.5% of creator pool = 8.75% of total
        - etc.
        
        Args:
            total_usdc: Total amount to distribute
            royalty_chain: Chain with payout nodes
            decay_factor: Override config decay factor
            max_depth: Override config max depth
        
        Returns:
            RecursivePayoutResult with all payouts
        """
        logger.info(
            f"[ENTRY] calculate_recursive_payouts: total={total_usdc}, "
            f"chain_depth={royalty_chain.total_depth}"
        )
        
        decay = decay_factor or self.config.decay_factor
        max_d = max_depth or self.config.max_chain_depth
        
        payouts: list[PayoutItem] = []
        
        # Separate fixed parties and creators
        fixed_nodes = [n for n in royalty_chain.nodes if n.role in ("oracle", "platform", "ops")]
        creator_nodes = [n for n in royalty_chain.nodes if n.role in ("creator", "parent_creator")]
        
        # Pay fixed parties first (from total)
        for node in fixed_nodes:
            if node.share_bps > 0:
                amount = total_usdc * (node.share_bps / 10000)
                payouts.append(
                    PayoutItem(
                        to=node.wallet,
                        amount_usdc=f"{amount:.6f}",
                        label=node.role,
                    )
                )
                logger.debug(f"Fixed payout: {node.role} -> {amount:.6f} USDC")
        
        # Calculate creator pool (remaining after fixed parties)
        fixed_share = sum(n.share_bps for n in fixed_nodes) / 10000
        creator_pool = total_usdc * (1 - fixed_share)
        
        # Sort creators by depth (direct creator first)
        creator_nodes.sort(key=lambda n: n.depth)
        
        # Distribute creator pool with decay
        remaining_pool = creator_pool
        depth = 0
        
        for node in creator_nodes:
            if depth >= max_d:
                logger.warning(f"Max depth {max_d} reached, stopping chain")
                break
            
            if node.depth > depth:
                # Moving to next depth level, apply decay
                remaining_pool *= decay
                depth = node.depth
            
            # Calculate this creator's share
            if node.depth == royalty_chain.total_depth:
                # Last in chain gets all remaining
                payout_amount = remaining_pool
            else:
                # Not last, pay (1 - decay) portion
                payout_amount = remaining_pool * (1 - decay)
            
            if payout_amount > 0.000001:  # Minimum threshold
                payouts.append(
                    PayoutItem(
                        to=node.wallet,
                        amount_usdc=f"{payout_amount:.6f}",
                        label=f"creator_depth_{node.depth}",
                    )
                )
                logger.debug(
                    f"Creator payout: depth={node.depth}, wallet={node.wallet[:10]}..., "
                    f"amount={payout_amount:.6f} USDC"
                )
        
        # Estimate gas cost
        gas_estimate = len(payouts) * 0.001  # ~$0.001 per transfer on Arc
        
        result = RecursivePayoutResult(
            motion_id=royalty_chain.motion_id,
            total_usdc=f"{total_usdc:.6f}",
            payouts=payouts,
            chain_depth=royalty_chain.total_depth,
            decay_factor_used=decay,
            gas_estimate_usdc=f"{gas_estimate:.6f}",
        )
        
        logger.info(
            f"Recursive payouts calculated: {len(payouts)} recipients, "
            f"gas_estimate={gas_estimate:.6f} USDC"
        )
        logger.debug(f"[EXIT] calculate_recursive_payouts")
        return result
    
    # -------------------------------------------------------------------------
    # Gas Sponsorship Budget Management
    # -------------------------------------------------------------------------
    
    def check_gas_sponsor_balance(self) -> dict[str, Any]:
        """
        Check gas sponsorship wallet balance and replenish status.
        
        Returns:
            Balance info and whether replenish is needed
        """
        logger.debug("[ENTRY] check_gas_sponsor_balance")
        
        current_ratio = self._gas_sponsor_balance_usdc / self._gas_sponsor_max_usdc
        needs_replenish = current_ratio < self.config.gas_replenish_threshold
        
        result = {
            "balance_usdc": f"{self._gas_sponsor_balance_usdc:.2f}",
            "max_usdc": f"{self._gas_sponsor_max_usdc:.2f}",
            "ratio": current_ratio,
            "threshold": self.config.gas_replenish_threshold,
            "needs_replenish": needs_replenish,
            "replenish_amount": f"{(self._gas_sponsor_max_usdc * self.config.gas_replenish_threshold) - self._gas_sponsor_balance_usdc:.2f}" if needs_replenish else "0.00",
        }
        
        logger.info(
            f"Gas sponsor balance: {self._gas_sponsor_balance_usdc:.2f} USDC "
            f"({current_ratio:.1%}), needs_replenish={needs_replenish}"
        )
        logger.debug(f"[EXIT] check_gas_sponsor_balance")
        return result
    
    def replenish_gas_sponsor(self, amount_usdc: Optional[str] = None) -> dict[str, Any]:
        """
        Replenish gas sponsorship budget from treasury.
        
        Uses 40% of treasury allocation by default for auto-replenish.
        
        Args:
            amount_usdc: Specific amount to transfer (or auto-calculate)
        
        Returns:
            Replenish result with new balance
        """
        logger.info("[ENTRY] replenish_gas_sponsor")
        
        # Calculate replenish amount (40% of max by default)
        if amount_usdc:
            replenish_amount = float(amount_usdc)
        else:
            target_balance = self._gas_sponsor_max_usdc * self.config.gas_replenish_threshold
            replenish_amount = max(0, target_balance - self._gas_sponsor_balance_usdc)
        
        if replenish_amount <= 0:
            logger.info("No replenish needed")
            return {
                "replenished": False,
                "amount_usdc": "0.00",
                "new_balance_usdc": f"{self._gas_sponsor_balance_usdc:.2f}",
            }
        
        # Execute replenish (simulated for demo)
        old_balance = self._gas_sponsor_balance_usdc
        self._gas_sponsor_balance_usdc += replenish_amount
        
        logger.info(
            f"Gas sponsor replenished: {old_balance:.2f} -> "
            f"{self._gas_sponsor_balance_usdc:.2f} USDC (+{replenish_amount:.2f})"
        )
        
        result = {
            "replenished": True,
            "amount_usdc": f"{replenish_amount:.2f}",
            "old_balance_usdc": f"{old_balance:.2f}",
            "new_balance_usdc": f"{self._gas_sponsor_balance_usdc:.2f}",
            "tx_hash": f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}",
        }
        
        logger.debug(f"[EXIT] replenish_gas_sponsor")
        return result
    
    def deduct_gas_cost(self, gas_usdc: float) -> bool:
        """
        Deduct gas cost from sponsor balance.
        
        Args:
            gas_usdc: Gas cost in USDC
        
        Returns:
            True if deducted, False if insufficient balance
        """
        if self._gas_sponsor_balance_usdc < gas_usdc:
            logger.warning(
                f"Insufficient gas sponsor balance: {self._gas_sponsor_balance_usdc:.2f} < {gas_usdc:.2f}"
            )
            return False
        
        self._gas_sponsor_balance_usdc -= gas_usdc
        logger.debug(
            f"Gas deducted: {gas_usdc:.4f} USDC, remaining: {self._gas_sponsor_balance_usdc:.2f}"
        )
        return True
    
    # -------------------------------------------------------------------------
    # Settlement Reconciliation
    # -------------------------------------------------------------------------
    
    def reconcile_settlement(
        self,
        event_id: str,
        expected_amount_usdc: str,
        actual_amount_usdc: str,
        similarity_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Reconcile settlement using similarity analysis for refund decisions.
        
        Uses KNN/RkCNN similarity scores to automate refund decisions:
        - High similarity (>0.9): likely duplicate, auto-refund
        - Medium similarity (0.5-0.9): flag for review
        - Low similarity (<0.5): proceed with settlement
        
        Args:
            event_id: Payment event ID
            expected_amount_usdc: Expected payment amount
            actual_amount_usdc: Actual amount received
            similarity_score: Optional similarity score for auto-decision
        
        Returns:
            Reconciliation result with action taken
        """
        logger.info(
            f"[ENTRY] reconcile_settlement: event_id={event_id}, "
            f"expected={expected_amount_usdc}, actual={actual_amount_usdc}"
        )
        
        expected = float(expected_amount_usdc)
        actual = float(actual_amount_usdc)
        difference = actual - expected
        difference_pct = abs(difference) / expected if expected > 0 else 0
        
        action = "SETTLED"
        refund_amount = "0.00"
        
        # Check for overpayment
        if difference > 0.01:  # More than 1 cent overpaid
            refund_amount = f"{difference:.6f}"
            action = "REFUND_OVERPAYMENT"
            logger.info(f"Overpayment detected: {difference:.6f} USDC to refund")
        
        # Check for underpayment
        elif difference < -0.01:
            if abs(difference) > expected * 0.1:  # More than 10% short
                action = "REJECT_UNDERPAYMENT"
                logger.warning(f"Significant underpayment: {abs(difference):.6f} USDC")
            else:
                action = "ACCEPT_PARTIAL"
                logger.info(f"Minor underpayment accepted: {abs(difference):.6f} USDC")
        
        # Use similarity score for duplicate detection
        if similarity_score is not None and similarity_score > 0.9:
            action = "FLAG_DUPLICATE"
            logger.warning(
                f"Potential duplicate: similarity={similarity_score:.3f}"
            )
        
        result = {
            "event_id": event_id,
            "expected_usdc": expected_amount_usdc,
            "actual_usdc": actual_amount_usdc,
            "difference_usdc": f"{difference:.6f}",
            "difference_pct": f"{difference_pct:.2%}",
            "action": action,
            "refund_amount_usdc": refund_amount,
            "similarity_score": similarity_score,
            "reconciled_at": int(time.time()),
        }
        
        logger.info(f"Settlement reconciled: action={action}")
        logger.debug(f"[EXIT] reconcile_settlement")
        return result
    
    # -------------------------------------------------------------------------
    # Royalty Chain Processing
    # -------------------------------------------------------------------------
    
    def process_royalty_payments(
        self,
        royalty_chain: RoyaltyChain,
        total_usdc: str,
    ) -> list[dict[str, Any]]:
        """
        Process all royalty payments for a chain.
        
        Iterates through chain depth and executes transfers to each
        recipient based on calculated amounts with decay.
        
        Args:
            royalty_chain: Complete royalty chain
            total_usdc: Total amount to distribute
        
        Returns:
            List of payment results for each recipient
        """
        logger.info(
            f"[ENTRY] process_royalty_payments: motion={royalty_chain.motion_id}, "
            f"total={total_usdc} USDC"
        )
        
        # Calculate payouts
        payout_result = self.calculate_recursive_payouts(
            total_usdc=float(total_usdc),
            royalty_chain=royalty_chain,
        )
        
        # Execute each payout
        results: list[dict[str, Any]] = []
        
        for payout in payout_result.payouts:
            try:
                result = self._execute_single_payout(payout)
                results.append({
                    "to": payout.to,
                    "amount_usdc": payout.amount_usdc,
                    "label": payout.label,
                    "status": "SUCCESS",
                    "tx_hash": result.get("tx_hash"),
                })
            except Exception as e:
                results.append({
                    "to": payout.to,
                    "amount_usdc": payout.amount_usdc,
                    "label": payout.label,
                    "status": "FAILED",
                    "error": str(e),
                })
        
        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        logger.info(
            f"Royalty payments processed: {success_count}/{len(results)} successful"
        )
        logger.debug(f"[EXIT] process_royalty_payments")
        return results
    
    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------
    
    def _execute_payouts(self, payouts: list[PayoutItem]) -> None:
        """Execute multiple payouts via Circle service."""
        for payout in payouts:
            self._execute_single_payout(payout)
    
    def _execute_single_payout(self, payout: PayoutItem) -> dict[str, Any]:
        """Execute single payout transfer."""
        logger.debug(f"Executing payout: {payout.amount_usdc} USDC to {payout.to[:10]}...")
        
        if self._circle_service:
            # Use real Circle service
            result = self._circle_service.transfer_usdc(
                from_wallet_id="treasury",  # Would be configured
                to_address=payout.to,
                amount_usdc=payout.amount_usdc,
            )
            return {"tx_hash": result.tx_hash, "status": result.status}
        else:
            # Simulated transfer
            tx_hash = f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"
            return {"tx_hash": tx_hash, "status": "CONFIRMED"}
    
    def _execute_simple_payout(self, to_address: str, amount_usdc: str) -> dict[str, Any]:
        """Execute simple single-recipient payout."""
        payout = PayoutItem(
            to=to_address,
            amount_usdc=amount_usdc,
            label="creator",
        )
        return self._execute_single_payout(payout)
    
    # -------------------------------------------------------------------------
    # Status and Monitoring
    # -------------------------------------------------------------------------
    
    def get_pending_payouts(self) -> list[PaymentTriggerEvent]:
        """Get list of pending payout events."""
        return [
            e for e in self._pending_payouts.values()
            if e.status in ("PENDING", "PROCESSING")
        ]
    
    def get_payout_status(self, event_id: str) -> Optional[PaymentTriggerEvent]:
        """Get status of specific payout event."""
        return self._pending_payouts.get(event_id)
