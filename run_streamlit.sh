#!/bin/bash

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "py310_venv" ]; then
    source py310_venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py --server.port=$STREAMLIT_SERVER_PORT --server.headless=$STREAMLIT_SERVER_HEADLESS --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS
