"""Explain the different types of scenarios."""
print("=" * 80)
print("TWO TYPES OF SCENARIOS")
print("=" * 80)

print("\n1. VALIDATION SCENARIOS (Race Conditions)")
print("-" * 80)
print("These test the engine under different race conditions:")
print()
scenarios = [
    ("base", "Normal race conditions (baseline)"),
    ("early_sc", "Early safety car scenario (SC in first 7 laps)"),
    ("late_sc", "Late safety car scenario (SC after lap 80%)"),
    ("hot_track", "High temperature (+7°C track temp)"),
    ("heavy_traffic", "High traffic density (traffic_density ≥ 3.0)"),
    ("undercut", "Undercut opportunity (2s gap ahead)"),
    ("no_weather", "No weather features (tests robustness)"),
]

print(f"{'Scenario':<20} {'Description':<60}")
print("-" * 80)
for name, desc in scenarios:
    print(f"{name:<20} {desc:<60}")

print("\n" + "=" * 80)
print("2. COUNTERFACTUAL SCENARIOS (Individual Vehicle Comparisons)")
print("-" * 80)
print("These are the 10 scenarios we showed earlier:")
print("  • Each compares: Recommended Strategy vs Actual Strategy")
print("  • For individual vehicles at specific laps")
print("  • Shows if the engine's recommendation would have been better/worse")
print()
print("Example:")
print("  Scenario 1: Vehicle GR86-002-2 at lap 1")
print("    • Engine recommended: Pit at lap 5")
print("    • Actual strategy: Different pit timing")
print("    • Result: +1.41s (recommended was WORSE)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("• Validation scenarios: 7 different race conditions to test")
print("• Counterfactual scenarios: 10 individual vehicle/lap comparisons")
print("• Currently only 'base' scenario has been run")
print("=" * 80)


