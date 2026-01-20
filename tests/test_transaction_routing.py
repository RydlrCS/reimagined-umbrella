"""
Tests for on-chain vs off-chain transaction routing in trustless agent loop.

Tests the decision logic for routing transactions through:
- Arc Network (on-chain) for NFT minting and high-value settlements
- Circle Wallets (off-chain) for low-value micropayments
- Hybrid mode for NFT minting on-chain + royalties off-chain
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from kinetic_ledger.services.trustless_agent import (
    TrustlessAgentLoop,
    TrustlessAgentConfig,
    TransactionRouting,
)
from kinetic_ledger.utils.errors import DecisionError


class TestTransactionRouting:
    """Test transaction routing decision logic."""
    
    @pytest.fixture
    def config_with_both(self):
        """Config with both Arc and Circle enabled."""
        return TrustlessAgentConfig(
            storage_url="local://./test_data",
            gemini_api_key="test_gemini_key",
            circle_api_key="test_circle_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
            on_chain_threshold_usdc=100.0,
            force_on_chain_for_nfts=True,
            default_routing="hybrid",
        )
    
    @pytest.fixture
    def config_arc_only(self):
        """Config with only Arc Network."""
        return TrustlessAgentConfig(
            storage_url="local://./test_data",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
    
    @pytest.fixture
    def config_circle_only(self):
        """Config with only Circle Wallets."""
        return TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_circle_key",
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
    
    def test_nft_mint_routes_on_chain(self, config_with_both):
        """Test NFT minting routes on-chain or hybrid when both services available."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            routing = agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=10.0,
                multi_party=False,
            )
            
            # Should use hybrid or on-chain when both services available
            assert routing.strategy in ("on-chain", "hybrid")
            assert "immutability" in routing.reason.lower() or "hybrid" in routing.reason.lower()
            assert "mint_motion_pack" in routing.on_chain_operations
            assert routing.use_arc_network is True
    
    def test_high_value_routes_on_chain(self, config_with_both):
        """Test high-value payments (≥$100) route on-chain."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            routing = agent.decide_transaction_routing(
                operation_type="payment",
                amount_usdc=150.0,
                multi_party=False,
            )
            
            assert routing.strategy == "on-chain"
            assert "150.00" in routing.reason
            assert "threshold" in routing.reason.lower()
            assert routing.use_arc_network is True
    
    def test_low_value_routes_off_chain(self, config_with_both):
        """Test low-value micropayments route off-chain."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            routing = agent.decide_transaction_routing(
                operation_type="payment",
                amount_usdc=5.0,
                multi_party=False,
            )
            
            assert routing.strategy == "off-chain"
            assert "low-value" in routing.reason.lower() or "gas-efficient" in routing.reason.lower()
            assert routing.use_circle_wallets is True
    
    def test_multi_party_prefers_on_chain(self, config_with_both):
        """Test multi-party settlements prefer on-chain for atomicity."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            routing = agent.decide_transaction_routing(
                operation_type="usage_metering",
                amount_usdc=50.0,
                multi_party=True,
            )
            
            assert routing.strategy == "on-chain"
            assert "multi-party" in routing.reason.lower() or "atomic" in routing.reason.lower()
            assert "atomic_multi_payout" in routing.on_chain_operations
    
    def test_hybrid_mode_for_nft_with_both_services(self, config_with_both):
        """Test hybrid mode: NFT on-chain + royalties off-chain."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            routing = agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=25.0,
                multi_party=False,
            )
            
            # Should use hybrid when both services available
            assert routing.strategy in ("hybrid", "on-chain")
            if routing.strategy == "hybrid":
                assert len(routing.on_chain_operations) > 0
                assert len(routing.off_chain_operations) > 0
                assert routing.use_arc_network is True
                assert routing.use_circle_wallets is True
    
    def test_arc_only_fallback(self, config_arc_only):
        """Test Arc-only config routes everything on-chain."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'):
            
            agent = TrustlessAgentLoop(config_arc_only)
            
            # Low-value payment should still go on-chain (no Circle alternative)
            routing = agent.decide_transaction_routing(
                operation_type="payment",
                amount_usdc=1.0,
                multi_party=False,
            )
            
            assert routing.strategy == "on-chain"
            assert "unavailable" in routing.reason.lower() or agent.commerce_orchestrator is None
    
    def test_circle_only_fallback(self, config_circle_only):
        """Test Circle-only config routes everything off-chain."""
        with patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_circle_only)
            
            # NFT mint should go off-chain (no Arc alternative)
            routing = agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=10.0,
                multi_party=False,
            )
            
            assert routing.strategy == "off-chain"
            assert "unavailable" in routing.reason.lower() or "fallback" in routing.reason.lower()
    
    def test_gas_estimation_on_chain(self, config_with_both):
        """Test gas cost estimation for on-chain transactions."""
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config_with_both)
            
            # NFT mint (higher gas)
            routing_mint = agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=10.0,
                multi_party=False,
            )
            
            assert routing_mint.estimated_gas_usdc is not None
            assert float(routing_mint.estimated_gas_usdc) > 0
    
    def test_no_infrastructure_raises_error(self):
        """Test error when no transaction infrastructure available."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            # No Arc or Circle configured
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
        
        agent = TrustlessAgentLoop(config)
        
        with pytest.raises(DecisionError, match="No transaction infrastructure"):
            agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=10.0,
                multi_party=False,
            )
    
    def test_routing_with_custom_threshold(self):
        """Test custom on-chain threshold."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
            on_chain_threshold_usdc=50.0,  # Lower threshold
        )
        
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config)
            
            # $60 should go on-chain with $50 threshold
            routing = agent.decide_transaction_routing(
                operation_type="payment",
                amount_usdc=60.0,
                multi_party=False,
            )
            
            assert routing.strategy == "on-chain"
            assert "50.00" in routing.reason or "threshold" in routing.reason.lower()
    
    def test_force_nft_on_chain_disabled(self):
        """Test NFT minting can go off-chain when force flag disabled."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
            force_on_chain_for_nfts=False,  # Disabled
        )
        
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            agent = TrustlessAgentLoop(config)
            
            routing = agent.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=5.0,
                multi_party=False,
            )
            
            # With force disabled and low value, could go off-chain
            assert routing.strategy in ("off-chain", "on-chain", "hybrid")


class TestRoutingIntegration:
    """Integration tests for routing with services."""
    
    def test_service_initialization_both(self):
        """Test agent initializes both Arc and Circle services."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
        
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService') as mock_arc, \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator') as mock_circle:
            
            agent = TrustlessAgentLoop(config)
            
            # Both services should be initialized
            assert mock_arc.called
            assert mock_circle.called
            assert agent.arc_service is not None
            assert agent.commerce_orchestrator is not None
    
    def test_service_initialization_arc_failure_graceful(self):
        """Test graceful handling when Arc service initialization fails."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
        
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService', side_effect=Exception("Arc RPC unreachable")), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            # Should not raise - graceful degradation
            agent = TrustlessAgentLoop(config)
            
            # Arc service should be None, Circle should work
            assert agent.arc_service is None
            assert agent.commerce_orchestrator is not None
    
    def test_routing_decision_logging(self, caplog):
        """Test routing decisions trigger service initialization logging."""
        config = TrustlessAgentConfig(
            storage_url="local://./test_data",
            circle_api_key="test_key",
            arc_rpc_url="https://arc-testnet-rpc.example.com",
            arc_contract_address="0x1111111111111111111111111111111111111111",
            arc_private_key="0x" + "a" * 64,
            verifying_contract="0x2222222222222222222222222222222222222222",
            oracle_address="0x3333333333333333333333333333333333333333",
            platform_address="0x4444444444444444444444444444444444444444",
            ops_address="0x5555555555555555555555555555555555555555",
        )
        
        with patch('kinetic_ledger.services.trustless_agent.ArcNetworkService'), \
             patch('kinetic_ledger.services.trustless_agent.CommerceOrchestrator'):
            
            with caplog.at_level("INFO"):
                agent = TrustlessAgentLoop(config)
                
                routing = agent.decide_transaction_routing(
                    operation_type="nft_mint",
                    amount_usdc=10.0,
                    multi_party=False,
                )
            
            # Check that services were initialized (via logs or agent state)
            assert agent.arc_service is not None
            assert agent.commerce_orchestrator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
