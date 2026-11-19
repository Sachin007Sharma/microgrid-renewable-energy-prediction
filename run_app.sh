#!/bin/bash
# run_app.sh — Launches the Microgrid Energy Forecaster web application
# Usage: bash run_app.sh

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ⚡  Microgrid Energy Forecaster"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Starting web server..."
echo "  Open your browser at: http://localhost:8501"
echo "  Press Ctrl+C to stop the server."
echo ""

cd "$(dirname "$0")"
/opt/anaconda3/bin/streamlit run app.py --server.port 8501 --server.headless false

