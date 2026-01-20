#!/usr/bin/env python3
"""
Quick test script for the Motion Visualizer API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_motion_library():
    """Test motion library endpoint"""
    print("Testing /api/motions/library...")
    response = requests.get(f"{BASE_URL}/api/motions/library")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total motions: {data.get('total', 0)}")
    for motion in data.get('motions', []):
        print(f"  - {motion['name']}: {motion['tags']}")
    print()

def test_prompt_analyze():
    """Test prompt analysis endpoint"""
    print("Testing /api/prompts/analyze...")
    payload = {
        "prompt": "Mix capoeira and breakdance smoothly, 70% breakdance"
    }
    response = requests.post(
        f"{BASE_URL}/api/prompts/analyze",
        json=payload
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_ui():
    """Test UI serving"""
    print("Testing UI...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Content length: {len(response.text)} chars")
    print(f"Title found: {'Kinetic Ledger' in response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Motion Visualizer API Test")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_ui()
        test_motion_library()
        test_prompt_analyze()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print()
        print(f"🎨 Open UI: {BASE_URL}")
        print()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
