"""
Quick test script to verify the AIOps system is working.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from main import AIOpsSystem
from memory.shared_state import SharedState, RepairMode
from agents.observability import ObservabilityAgent
from agents.detection import DetectionAgent
from memory.shared_state import ObservabilityData, AnomalyEvent

print("=" * 60)
print("RCAgentX Quick Test")
print("=" * 60)

# Test 1: Initialize system
print("\n[1/3] Initializing system...")
try:
    aiops = AIOpsSystem.from_env()
    print("      System initialized successfully!")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# Test 2: Test individual agents
print("\n[2/3] Testing individual agents...")

# Create mock observability data
obs_data = ObservabilityData(
    metrics=[
        {"query": "cpu_usage", "data": [{"metric": {"pod": "test"}, "values": [[1, 0.95]]}]}
    ],
    logs=[
        {"line": "ERROR: Connection timeout", "timestamp": "2024-01-01T00:00:00"}
    ]
)

# Test Detection Agent
detection_agent = DetectionAgent(verbose=False)
state = {
    "observability": obs_data,
    "logs": [],
    "errors": []
}

try:
    result = detection_agent.execute(state)
    anomaly = result.get("anomaly")
    if anomaly:
        print(f"      Detection: Found {anomaly.type} anomaly (severity: {anomaly.severity})")
    else:
        print("      Detection: No anomaly detected")
except Exception as e:
    print(f"      Detection FAILED: {e}")

# Test 3: Knowledge base
print("\n[3/3] Testing knowledge base...")
try:
    stats = aiops.get_knowledge_base_stats()
    print(f"      Knowledge base: {stats['total_records']} records, {stats['success_rate']:.1%} success rate")
except Exception as e:
    print(f"      Knowledge base test: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)

# Run full workflow if requested
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Run full workflow test")
args = parser.parse_args()

if args.full:
    print("\nRunning full workflow test...")
    result = aiops.process_incident(
        alert_labels={"service": "api-service", "severity": "high"},
        entities=["api-service-pod-1"]
    )
    print(f"\nFinal status: {result.get('status')}")
    if result.get("report"):
        print(f"Report summary: {result['report'].get('summary', 'N/A')[:200]}")
