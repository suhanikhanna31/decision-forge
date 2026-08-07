"""
Test: Make one decision with AI explanation.
"""
from app.config_loader import load_config
from app.ai.ai_enhanced_engine import AIEnhancedDecisionEngine

print("=" * 70)
print("Testing AI Decision with Explanation")
print("=" * 70)

# Initialize
config = load_config("configs/ecommerce.yaml")
engine = AIEnhancedDecisionEngine(config, enable_ai=True)

# Test input - a high-risk customer
user_input = {
    "anomaly_score": 0.82,
    "request_count_today": 15,
    "expected_lift": 0.15,
    "revenue": 1200,
    "incentive_cost": 150,
    "churn": 0.75
}

print("\n📊 User Metrics:")
for key, value in user_input.items():
    print(f"   {key}: {value}")

# Make decision WITHOUT AI first (baseline)
print("\n1️⃣ BASELINE DECISION (No AI):")
baseline = engine.base_engine.decide(user_input)
print(f"   Decision: {baseline['decision']}")
print(f"   Reason: {baseline['reason']}")
print(f"   Expected Value: ${baseline.get('expected_value', 0):.2f}")

# Make decision WITH AI explanation
print("\n2️⃣ AI-ENHANCED DECISION:")
print("   (Generating AI explanation... may take 5-10 seconds)")
print()

result = engine.decide_with_explanation(
    inputs=user_input,
    return_explanation=True,
    return_recommendations=True
)

print(f"   Decision: {result['decision']['decision']}")
print(f"   Reason: {result['decision']['reason']}")
print(f"   Expected Value: ${result['decision'].get('expected_value', 0):.2f}")

print(f"\n   📝 AI Explanation:")
print(f"   {result['explanation']}")

if 'recommendations' in result:
    print(f"\n   💡 AI Recommendations:")
    recs = result['recommendations']
    print(f"   Summary: {recs.get('summary', 'N/A')}")
    print(f"   Rationale: {recs.get('rationale', 'N/A')}")
    print(f"   Actions: {recs.get('suggested_actions', 'N/A')}")

print("\n" + "=" * 70)
print("✅ Test completed successfully!")
print("=" * 70)