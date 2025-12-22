#!/usr/bin/env python3
"""Quick start script for Graph Self-Supervised Learning."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="Quick start for Graph SSL")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--demo-only", action="store_true", help="Only run the demo")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    print("🚀 Graph Self-Supervised Learning Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not args.skip_install and not args.test_only:
        if not run_command("pip install -r requirements.txt", "Installing dependencies"):
            sys.exit(1)
    
    # Run tests
    if not args.demo_only:
        if not run_command("pytest tests/ -v", "Running tests"):
            print("⚠️  Some tests failed, but continuing...")
    
    # Quick training example
    if not args.demo_only and not args.test_only:
        print("\n🏋️  Running quick training example...")
        
        # Create a simple training command
        train_cmd = (
            "python -m src.train.main "
            "--dataset synthetic_sbm_1000 "
            "--model node_masking "
            "--epochs 10 "
            "--output outputs/quick_start"
        )
        
        if not run_command(train_cmd, "Quick training example"):
            print("⚠️  Training failed, but continuing...")
    
    # Launch demo
    if not args.test_only:
        print("\n🎮 Launching interactive demo...")
        print("The demo will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the demo")
        
        try:
            subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed to start: {e}")
            print("Make sure Streamlit is installed: pip install streamlit")
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Explore the demo: streamlit run demo/app.py")
    print("2. Train your own models: python -m src.train.main --help")
    print("3. Read the documentation: README.md")


if __name__ == "__main__":
    main()
