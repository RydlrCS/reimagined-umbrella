# API Key Security Best Practices

## Overview

This document outlines security best practices for managing Gemini API keys in the Kinetic Ledger project, following [Gemini API official guidelines](https://ai.google.dev/gemini-api/docs/api-key).

## Critical Security Rules

### 1. Keep API Keys Confidential

**✅ DO:**
- Store API keys in environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)
- Use `.env` files locally (excluded from git via `.gitignore`)
- Configure environment variables in production (Kubernetes secrets, Cloud Run env vars)

**❌ DON'T:**
- Commit API keys to source control (Git, GitHub)
- Hardcode API keys in source code
- Expose API keys in client-side code (web browsers, mobile apps)
- Share API keys in Slack, email, or documentation

### 2. Environment Variable Setup

The Kinetic Ledger codebase automatically detects `GEMINI_API_KEY` when set as an environment variable.

#### Linux/macOS (Bash)

```bash
# Add to ~/.bashrc or ~/.bash_profile
export GEMINI_API_KEY="your_actual_api_key_here"

# Apply changes
source ~/.bashrc
```

#### macOS (Zsh)

```bash
# Add to ~/.zshrc
export GEMINI_API_KEY="your_actual_api_key_here"

# Apply changes
source ~/.zshrc
```

#### Windows

1. Search for "Environment Variables" in Start menu
2. Edit **System Settings** → **Environment Variables**
3. Add new variable:
   - Name: `GEMINI_API_KEY`
   - Value: `your_actual_api_key_here`
4. Restart terminal

### 3. Local Development

**Create a `.env` file** (already gitignored):

```bash
# Copy example and edit with your real key
cp .env.example .env

# Edit .env with your API key
GEMINI_API_KEY=AIzaSy...your_real_key_here

# Load environment variables (if using python-dotenv)
# Code automatically picks up GEMINI_API_KEY
```

**Verify `.env` is gitignored:**

```bash
git status  # Should NOT show .env as untracked
cat .gitignore | grep .env  # Should show .env pattern
```

### 4. Production Deployment

#### Kubernetes (Recommended)

```yaml
# Create secret
kubectl create secret generic gemini-api-key \
  --from-literal=GEMINI_API_KEY=your_api_key_here \
  -n kinetic-ledger

# Reference in deployment
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: attestation-oracle
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gemini-api-key
              key: GEMINI_API_KEY
```

#### Google Cloud Run

```bash
gcloud run deploy attestation-oracle \
  --set-env-vars="GEMINI_API_KEY=your_api_key_here" \
  --region=us-central1
```

#### Docker Compose

```yaml
services:
  attestation-oracle:
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}  # Passed from host
    env_file:
      - .env  # Load from .env file (local only)
```

## Code Usage Patterns

### ✅ Correct Pattern (Environment Variable)

```python
from kinetic_ledger.connectors import FileSearchConnector
from kinetic_ledger.services import EmbeddingService
import os

# Automatic detection from environment
connector = FileSearchConnector()  # Uses GEMINI_API_KEY automatically
embedding = EmbeddingService()     # Uses GEMINI_API_KEY automatically

# Explicit environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

connector = FileSearchConnector(api_key=api_key)
```

### ❌ Incorrect Pattern (Hardcoded)

```python
# NEVER DO THIS - Hardcoded API key
connector = FileSearchConnector(
    api_key="AIzaSyCHVrm2t4bHlgIta5GHm8CftC9zZYQwIJM"  # ❌ SECURITY RISK
)
```

### ✅ Testing Without API Key

```python
# Graceful degradation when API key unavailable
connector = FileSearchConnector()
if not connector.is_available():
    # Use fallback: embedding cache only
    logger.warning("File Search unavailable - using embedding cache")

embedding = EmbeddingService()
vectors = embedding.generate_embeddings(text)  # Uses fallback if no API key
```

## API Key Restrictions (Optional)

For enhanced security, restrict your API key in [Google AI Studio](https://aistudio.google.com/app/apikey):

### Application Restrictions
- **HTTP referrers**: Limit to specific domains
- **IP addresses**: Limit to production server IPs
- **Android apps**: Restrict to app package name + SHA-1
- **iOS apps**: Restrict to bundle ID

### API Restrictions
- Enable **only** Generative Language API
- Disable unused APIs to minimize attack surface

## Audit & Rotation

### Regular Audits
1. Review active API keys monthly in [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Delete unused keys immediately
3. Check usage quotas for unexpected spikes
4. Monitor billing for anomalies

### Key Rotation
Rotate API keys every 90 days:

```bash
# 1. Create new API key in Google AI Studio
# 2. Update production environment variables
kubectl set env deployment/attestation-oracle \
  GEMINI_API_KEY=new_key_here \
  -n kinetic-ledger

# 3. Verify new key works
kubectl logs deployment/attestation-oracle -n kinetic-ledger

# 4. Delete old key in Google AI Studio
```

## What to Do if Key is Compromised

**Immediate Actions:**

1. **Revoke the key** in [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Generate new API key**
3. **Update all production environments** with new key
4. **Review usage logs** for unauthorized activity
5. **Audit recent commits** if key was committed to Git
6. **Force push history rewrite** if key is in Git history:

```bash
# Remove key from Git history (use with caution)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push to remote
git push origin --force --all
```

## Testing Best Practices

### ✅ Secure Test Pattern

All tests in this repository follow these security patterns:

```python
import os
import pytest

# SECURITY: Only load from environment, never hardcode
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HAS_API_KEY = bool(GEMINI_API_KEY)

# Skip entire test class if API key not available
@pytest.mark.skipif(
    not HAS_API_KEY,
    reason="GEMINI_API_KEY not set - skipping integration tests"
)
class TestGeminiIntegration:
    
    def test_client_initialization(self):
        client = GeminiClient(api_key=GEMINI_API_KEY)
        
        # ✅ CORRECT: Check key exists without exposing value
        assert client.api_key is not None
        assert len(client.api_key) > 0
        
        # ❌ WRONG: Would expose key in test output
        # assert client.api_key == GEMINI_API_KEY  # DON'T DO THIS
        # print(f"API Key: {client.api_key}")      # DON'T DO THIS
```

### Common Test Security Pitfalls

**❌ DON'T:**
- Assert equality with actual API key (`assert api_key == GEMINI_API_KEY`)
- Print API keys in test output (`print(f"Key: {api_key}")`)
- Include API keys in test fixtures
- Store API keys in test data files
- Log API keys in test error messages
- Commit `.env` files with real keys

**✅ DO:**
- Skip tests when API key unavailable (`@pytest.mark.skipif`)
- Assert key existence only (`assert api_key is not None`)
- Use environment variables exclusively
- Mock API responses for unit tests
- Test graceful degradation without API key

## CI/CD Security

### GitHub Actions (Example)

**❌ INSECURE - Logs expose keys:**
```yaml
- name: Run tests
  run: |
    export GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }}
    pytest -v  # ❌ Verbose output may print API key
```

**✅ SECURE - Keys never logged:**
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests (with API key from secrets)
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          # Tests automatically skip if key not set
          pytest tests/ -v --tb=line
          
      - name: Run tests (without API key - fallback mode)
        run: |
          # Unset to test graceful degradation
          unset GEMINI_API_KEY
          pytest tests/ -v -m "not requires_api_key"
```

**Security Notes:**
- Store API key in GitHub Secrets (Settings → Secrets → Actions)
- Use `env:` block to inject secrets (not inline)
- Enable "Require approval for all outside collaborators" for PRs
- Don't print environment variables in logs (`env`, `printenv`)
- Mask sensitive output with `::add-mask::` if needed

### GitLab CI/CD

```yaml
test:
  script:
    - pip install -e ".[dev]"
    - pytest tests/ -v --tb=line
  variables:
    GEMINI_API_KEY: $CI_GEMINI_API_KEY  # From GitLab CI/CD Variables
  only:
    - main
    - merge_requests
```

## Security Checklist

Before committing code or deploying, verify:

### Development
- [ ] No hardcoded API keys in source code
- [ ] `.env` file in `.gitignore`
- [ ] All API keys loaded from environment variables only
- [ ] Test assertions never compare against actual API key value
- [ ] No `print()` or `logging` statements that could expose keys
- [ ] Git history clean (no accidentally committed keys)

### Testing
- [ ] Integration tests skip when `GEMINI_API_KEY` not set
- [ ] Tests use `@pytest.mark.skipif(not HAS_API_KEY, ...)`
- [ ] No assertions like `assert api_key == GEMINI_API_KEY`
- [ ] No API keys in test fixtures or data files
- [ ] Graceful degradation tests without API key

### CI/CD
- [ ] API keys stored in CI/CD secrets (GitHub Actions, GitLab CI)
- [ ] Secrets injected via `env:` block, not inline
- [ ] No verbose logging that could expose keys (`-v` with caution)
- [ ] PR builds from forks don't access production keys
- [ ] Failed test logs don't contain API key values

### Production
- [ ] API keys in Kubernetes secrets or Cloud provider secrets manager
- [ ] Environment variables set at runtime (not baked into images)
- [ ] API key restrictions enabled (IP, HTTP referrer, or app)
- [ ] Only Generative Language API enabled
- [ ] Monitoring and alerting for unusual usage
- [ ] Key rotation schedule established (90 days)

### Repository (Public Repos)
- [ ] No `.env` files committed
- [ ] `.env.example` contains only placeholders
- [ ] README doesn't contain API keys
- [ ] Documentation uses `your_api_key_here` placeholders
- [ ] Git history scanned for accidentally committed keys
- [ ] Dependency audit for packages that might log env vars

## Current Status

✅ **Secure Patterns:**
- All services use environment variables
- `.env` excluded from Git via `.gitignore`
- Graceful degradation when API key unavailable
- No hardcoded keys in source code (as of latest update)

⚠️ **Action Required:**
- Set `GEMINI_API_KEY` environment variable for testing
- Obtain valid API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Configure production secrets (Kubernetes, Cloud Run, etc.)

## References

- [Gemini API Key Documentation](https://ai.google.dev/gemini-api/docs/api-key)
- [API Key Restrictions Guide](https://cloud.google.com/api-keys/docs/add-restrictions-api-keys)
- [Google Cloud Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Best Practices for API Keys](https://support.google.com/googleapi/answer/6310037)

---

**Last Updated:** January 9, 2026  
**Compliance:** Follows Gemini API [Terms of Service](https://ai.google.dev/gemini-api/terms)
