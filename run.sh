#!/bin/bash
# Quick start script for RCAgentX

set -e

# Activate virtual environment
source .venv/bin/activate

# Run the main script with arguments
python main.py "$@"
