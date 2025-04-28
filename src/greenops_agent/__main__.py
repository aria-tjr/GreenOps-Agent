"""
Main entry point for GreenOps Agent.

This allows the package to be run directly via:
    python -m greenops_agent [command]
"""
import sys
from greenops_agent.main import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())