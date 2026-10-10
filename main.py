"""
main.py
Root entrypoint for the Helicopter Engine Health & RUL Pipeline.

Usage:
    python main.py             # Run complete pipeline end-to-end
    streamlit run app.py       # Launch interactive web dashboard
"""
import sys
from pathlib import Path

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from run_pipeline import main

if __name__ == "__main__":
    main()
