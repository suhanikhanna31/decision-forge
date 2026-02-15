"""
Test Ollama AI integration.
"""
import sys

print("=" * 60)
print("Testing AI Setup (Ollama - FREE)")
print("=" * 60)

# Test 1: Python version
print(f"\n1. Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("   ✓ Version OK")

# Test 2: Ollama
print("\n2. Testing Ollama...")
try:
    import ollama
    models = ollama.list()
    print(f"   ✓ Ollama connected")
    print(f"   Available models: {len(models.get('models', []))}")
except Exception as e:
    print(f"   ❌ Ollama not running: {e}")
    print("   Run in another terminal: ollama serve")
    exit(1)

# Test 3: Imports
print("\n3. Testing imports...")
try:
    from app.config_loader import load_config
    print("   ✓ config_loader")
    from app.ai.ai_explainer import AIExplainer
    print("   ✓ ai_explainer")
    from app.ai.ai_enhanced_engine import AIEnhancedDecisionEngine
    print("   ✓ ai_enhanced_engine")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 4: Integration
print("\n4. Testing DecisionForge integration...")
try:
    config = load_config("configs/ecommerce.yaml")
    engine = AIEnhancedDecisionEngine(config, enable_ai=True)
    print(f"   ✓ Engine initialized")
    print(f"   AI enabled: {engine.enable_ai}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\nNext: python test_simple_decision.py")