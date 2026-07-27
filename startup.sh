#!/bin/bash
# Startup script for Azure App Service (Linux Python runtime)

echo "Installing required system packages (OpenCV dependencies)..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

echo "Starting Health Sight AI Streamlit App..."
python -m streamlit run app/demo_app.py --server.port=8000 --server.address=0.0.0.0
