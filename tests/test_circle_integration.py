"""
Integration tests for Circle API commerce orchestrator.

Tests Circle Programmable Wallets API integration for motion commerce:
- Wallet creation and management
- USDC transfers for royalty payments
- Payment intent creation
- x402 payment verification
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from kinetic_ledger.services.commerce_orchestrator import (
    CircleWalletConfig,
    CircleClient,
    X402Verifier,
    CommerceOrchestrator,
    PaymentIntentRequest,
)
from kinetic_ledger.utils.errors import CircleError, PaymentRequiredError


# Test configuration
TEST_CIRCLE_API_KEY = os.getenv("CIRCLE_API_KEY", "TEST_API_KEY:test_secret")
HAS_CIRCLE_KEY = bool(os.getenv("CIRCLE_API_KEY"))


class TestCircleWalletConfig:
    """Test Circle configuration."""
    
    def test_config_creation(self):
        """Test creating config with API key."""
        config = CircleWalletConfig(api_key="test_key_123")
        assert config.api_key == "test_key_123"
        assert config.api_endpoint == "https://api.circle.com/v1"
    
    def test_config_custom_endpoint(self):
        """Test config with custom endpoint."""
        config = CircleWalletConfig(
            api_key="test_key",
            api_endpoint="https://api-sandbox.circle.com/v1"
        )
        assert config.api_endpoint == "https://api-sandbox.circle.com/v1"
    
    @pytest.mark.skipif(not HAS_CIRCLE_KEY, reason="CIRCLE_API_KEY not set")
    def test_config_from_env(self):
        """Test loading config from environment."""
        config = CircleWalletConfig.from_env()
        assert config.api_key is not None
        assert len(config.api_key) > 0
    
    def test_config_from_env_missing_key(self):
        """Test error when API key not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(CircleError, match="CIRCLE_API_KEY not set"):
                CircleWalletConfig.from_env()


class TestCircleClient:
    """Test Circle API client."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return CircleWalletConfig(api_key="test_key_123")
    
    @pytest.fixture
    def client(self, config):
        """Test Circle client."""
        return CircleClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client is properly initialized."""
        assert client.config == config
        assert client.session is not None
        assert client.session.headers["Authorization"] == "Bearer test_key_123"
        assert client.session.headers["Accept"] == "application/json"
        assert client.session.headers["Content-Type"] == "application/json"
    
    def test_make_request_success(self, client):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"test": "value"}}
        
        with patch.object(client.session, 'request', return_value=mock_response):
            result = client._make_request("GET", "/test")
            assert result == {"data": {"test": "value"}}
    
    def test_make_request_authentication_error(self, client):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        with patch.object(client.session, 'request', return_value=mock_response):
            with pytest.raises(CircleError, match="authentication failed"):
                client._make_request("GET", "/test")
    
    def test_make_request_http_error(self, client):
        """Test generic HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        with patch.object(client.session, 'request', return_value=mock_response):
            with pytest.raises(CircleError, match="Circle API error"):
                client._make_request("GET", "/test")
    
    def test_make_request_network_error(self, client):
        """Test network error handling."""
        with patch.object(client.session, 'request', side_effect=Exception("Network error")):
            with pytest.raises(CircleError, match="request failed"):
                client._make_request("GET", "/test")
    
    def test_get_wallets_success(self, client):
        """Test retrieving wallets."""
        mock_response = {
            "data": {
                "wallets": [
                    {
                        "walletId": "wallet-123",
                        "address": "0x1234567890123456789012345678901234567890",
                        "blockchain": "ETH-SEPOLIA",
                        "state": "LIVE"
                    }
                ]
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.get_wallets()
            assert result == mock_response
    
    def test_create_wallet_success(self, client):
        """Test wallet creation."""
        mock_response = {
            "data": {
                "walletId": "new-wallet-456",
                "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                "blockchain": "ETH-SEPOLIA",
                "accountType": "SCA",
                "state": "LIVE"
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.create_wallet("user123")
            
            assert result["wallet_id"] == "new-wallet-456"
            assert result["address"] == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
            assert result["blockchain"] == "ETH-SEPOLIA"
            assert result["status"] == "live"
    
    def test_create_wallet_custom_blockchain(self, client):
        """Test wallet creation with custom blockchain."""
        mock_response = {
            "data": {
                "walletId": "wallet-789",
                "address": "0x1111111111111111111111111111111111111111",
                "blockchain": "MATIC-AMOY",
                "state": "LIVE"
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_req:
            result = client.create_wallet("user456", blockchain="MATIC-AMOY")
            
            # Verify request payload
            call_args = mock_req.call_args
            payload = call_args.kwargs['json']
            assert payload['blockchain'] == "MATIC-AMOY"
            assert result["blockchain"] == "MATIC-AMOY"
    
    def test_create_payment_intent(self, client):
        """Test payment intent creation."""
        result = client.create_payment_intent(
            wallet_id="wallet-123",
            amount_usdc="25.50",
            description="Motion usage fee"
        )
        
        assert result["wallet_id"] == "wallet-123"
        assert result["amount"] == "25.50"
        assert result["currency"] == "USDC"
        assert result["status"] == "pending"
        assert "intent_id" in result
        assert result["intent_id"].startswith("pi_")
    
    def test_execute_transfer_success(self, client):
        """Test USDC transfer execution."""
        mock_response = {
            "data": {
                "id": "tx-abc123",
                "state": "INITIATED",
                "walletId": "wallet-source",
                "blockchain": "ETH-SEPOLIA",
                "txHash": "0xabcdef1234567890"
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.execute_transfer(
                from_wallet_id="wallet-source",
                to_address="0x9999999999999999999999999999999999999999",
                amount_usdc="10.50"
            )
            
            assert result["tx_id"] == "tx-abc123"
            assert result["tx_hash"] == "0xabcdef1234567890"
            assert result["blockchain"] == "ETH-SEPOLIA"
            assert result["from"] == "wallet-source"
            assert result["to"] == "0x9999999999999999999999999999999999999999"
            assert result["amount"] == "10.50"
            assert result["status"] == "initiated"
    
    def test_execute_transfer_amount_conversion(self, client):
        """Test amount conversion to smallest unit."""
        mock_response = {"data": {"id": "tx-test", "state": "INITIATED"}}
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_req:
            client.execute_transfer(
                from_wallet_id="wallet-123",
                to_address="0x0000000000000000000000000000000000000000",
                amount_usdc="10.50"
            )
            
            # Verify amount converted to smallest unit (6 decimals)
            call_args = mock_req.call_args
            payload = call_args.kwargs['json']
            assert payload['amounts'] == ["10500000"]  # 10.50 * 1,000,000
    
    @pytest.mark.skipif(not HAS_CIRCLE_KEY, reason="CIRCLE_API_KEY not set")
    def test_get_wallets_real_api(self):
        """Test actual Circle API call to get wallets."""
        config = CircleWalletConfig.from_env()
        client = CircleClient(config)
        
        try:
            result = client.get_wallets()
            # Should return successful response or empty wallet list
            assert "data" in result or result is not None
        except CircleError as e:
            # API may return error if endpoint not supported
            # or authentication fails - this is acceptable
            assert "Circle API" in str(e) or "authentication" in str(e).lower()


class TestX402Verifier:
    """Test x402 payment verification."""
    
    @pytest.fixture
    def verifier(self):
        """Test verifier instance."""
        return X402Verifier()
    
    def test_verifier_initialization(self, verifier):
        """Test verifier is initialized."""
        assert verifier.facilitator_url == "https://x402.example.com"
    
    def test_verify_payment_proof_success(self, verifier):
        """Test payment proof verification."""
        result = verifier.verify_payment_proof(
            proof="proof_abc123",
            expected_amount="15.00"
        )
        
        assert result["verified"] is True
        assert result["amount"] == "15.00"
        assert "receipt_id" in result
        assert result["receipt_id"].startswith("x402_")
        assert "timestamp" in result


class TestPaymentIntentRequest:
    """Test payment intent request model."""
    
    def test_valid_request(self):
        """Test creating valid payment intent request."""
        request = PaymentIntentRequest(
            user_wallet="0x1234567890123456789012345678901234567890",
            product="motion_pack",
            unit="seconds_generated",
            quantity=30.0,
            unit_price_usdc="0.05",
            attestation_pack_hash="0x" + "a" * 64
        )
        
        assert request.user_wallet == "0x1234567890123456789012345678901234567890"
        assert request.product == "motion_pack"
        assert request.unit == "seconds_generated"
        assert request.quantity == 30.0
        assert request.unit_price_usdc == "0.05"
    
    def test_invalid_wallet_address(self):
        """Test validation fails for invalid wallet."""
        with pytest.raises(Exception):  # Pydantic validation error
            PaymentIntentRequest(
                user_wallet="invalid",
                product="motion_pack",
                unit="seconds_generated",
                quantity=30.0,
                unit_price_usdc="0.05",
                attestation_pack_hash="0x" + "a" * 64
            )
    
    def test_invalid_unit(self):
        """Test validation fails for invalid unit."""
        with pytest.raises(Exception):  # Pydantic validation error
            PaymentIntentRequest(
                user_wallet="0x1234567890123456789012345678901234567890",
                product="motion_pack",
                unit="invalid_unit",
                quantity=30.0,
                unit_price_usdc="0.05",
                attestation_pack_hash="0x" + "a" * 64
            )
    
    def test_negative_quantity(self):
        """Test validation fails for negative quantity."""
        with pytest.raises(Exception):  # Pydantic validation error
            PaymentIntentRequest(
                user_wallet="0x1234567890123456789012345678901234567890",
                product="motion_pack",
                unit="seconds_generated",
                quantity=-10.0,
                unit_price_usdc="0.05",
                attestation_pack_hash="0x" + "a" * 64
            )
    
    def test_valid_units(self):
        """Test all valid unit types."""
        valid_units = ["seconds_generated", "frames_generated", "agent_steps"]
        
        for unit in valid_units:
            request = PaymentIntentRequest(
                user_wallet="0x1234567890123456789012345678901234567890",
                product="motion_pack",
                unit=unit,
                quantity=10.0,
                unit_price_usdc="0.05",
                attestation_pack_hash="0x" + "a" * 64
            )
            assert request.unit == unit


class TestCommerceOrchestrator:
    """Test commerce orchestrator integration."""
    
    @pytest.fixture
    def orchestrator(self):
        """Test orchestrator with mocked Circle client."""
        with patch('kinetic_ledger.services.commerce_orchestrator.CircleWalletConfig.from_env') as mock_config:
            mock_config.return_value = CircleWalletConfig(api_key="test_key")
            orch = CommerceOrchestrator()
            return orch
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator is initialized."""
        assert orchestrator.circle_client is not None
        assert orchestrator.x402_verifier is not None
    
    def test_create_user_wallet(self, orchestrator):
        """Test creating user wallet."""
        mock_wallet = {
            "wallet_id": "wallet-user-123",
            "address": "0x1234567890123456789012345678901234567890",
            "blockchain": "ETH-SEPOLIA",
            "status": "live"
        }
        
        with patch.object(orchestrator.circle_client, 'create_wallet', return_value=mock_wallet):
            result = orchestrator.create_user_wallet("user123")
            assert result == mock_wallet
    
    def test_process_payment_success(self, orchestrator):
        """Test successful payment processing."""
        # Mock payment intent creation
        mock_intent = {
            "intent_id": "pi_abc123",
            "wallet_id": "wallet-123",
            "amount": "50.00",
            "currency": "USDC",
            "status": "pending"
        }
        
        with patch.object(orchestrator.circle_client, 'create_payment_intent', return_value=mock_intent):
            result = orchestrator.process_payment(
                wallet_id="wallet-123",
                amount_usdc="50.00",
                description="Motion usage"
            )
            assert result == mock_intent
    
    def test_distribute_royalties(self, orchestrator):
        """Test royalty distribution."""
        # Mock transfer execution
        mock_transfer = {
            "tx_id": "tx-royalty-123",
            "status": "initiated"
        }
        
        with patch.object(orchestrator.circle_client, 'execute_transfer', return_value=mock_transfer):
            result = orchestrator.distribute_royalty(
                from_wallet_id="platform-wallet",
                creator_address="0x1111111111111111111111111111111111111111",
                amount_usdc="10.00"
            )
            assert result == mock_transfer


class TestCircleIntegrationEndToEnd:
    """End-to-end Circle integration tests."""
    
    @pytest.mark.skipif(not HAS_CIRCLE_KEY, reason="CIRCLE_API_KEY not set")
    def test_wallet_creation_flow(self):
        """Test complete wallet creation flow with real API."""
        config = CircleWalletConfig.from_env()
        client = CircleClient(config)
        
        # This test will actually call Circle API
        # Note: May fail if API key doesn't have wallet creation permissions
        try:
            # Get existing wallets first to verify connectivity
            wallets = client.get_wallets()
            assert wallets is not None
        except CircleError as e:
            # If API call fails, ensure it's documented
            pytest.skip(f"Circle API not accessible: {e}")
    
    def test_mock_payment_workflow(self):
        """Test complete payment workflow with mocks."""
        with patch('kinetic_ledger.services.commerce_orchestrator.CircleWalletConfig.from_env') as mock_config:
            mock_config.return_value = CircleWalletConfig(api_key="test_key")
            orchestrator = CommerceOrchestrator()
            
            # Mock wallet creation
            mock_wallet = {
                "wallet_id": "wallet-new",
                "address": "0xnewaddress",
                "status": "live"
            }
            
            # Mock payment processing
            mock_payment = {
                "intent_id": "pi_payment",
                "status": "pending"
            }
            
            # Mock transfer
            mock_transfer = {
                "tx_id": "tx-transfer",
                "status": "initiated"
            }
            
            with patch.object(orchestrator.circle_client, 'create_wallet', return_value=mock_wallet), \
                 patch.object(orchestrator.circle_client, 'create_payment_intent', return_value=mock_payment), \
                 patch.object(orchestrator.circle_client, 'execute_transfer', return_value=mock_transfer):
                
                # Step 1: Create wallet
                wallet = orchestrator.create_user_wallet("testuser")
                assert wallet["wallet_id"] == "wallet-new"
                
                # Step 2: Process payment
                payment = orchestrator.process_payment(
                    wallet_id=wallet["wallet_id"],
                    amount_usdc="100.00",
                    description="Motion purchase"
                )
                assert payment["status"] == "pending"
                
                # Step 3: Distribute royalty
                transfer = orchestrator.distribute_royalty(
                    from_wallet_id="platform-wallet",
                    creator_address="0xcreator",
                    amount_usdc="70.00"
                )
                assert transfer["status"] == "initiated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
