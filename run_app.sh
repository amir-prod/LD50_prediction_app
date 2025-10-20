#!/bin/bash

# Script to run the Toxicity Prediction Streamlit App

echo "🧪 Starting Toxicity Prediction Web App..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "📦 Installing required packages..."
    pip install -r requirements.txt
fi

# Run the streamlit app
echo "Launching app..."
streamlit run app.py

