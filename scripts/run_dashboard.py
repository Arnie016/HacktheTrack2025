#!/usr/bin/env python3
"""Launch the web dashboard."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

if __name__ == "__main__":
    print("🚀 Starting GR Cup Strategy Dashboard...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)

