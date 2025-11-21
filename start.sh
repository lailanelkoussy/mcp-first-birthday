#!/bin/bash

echo "Starting Python application..."
echo "Binding to 0.0.0.0:7860"
python -u gradio_app.py --host 0.0.0.0 --port 7860 --repo-url https://github.com/lailanelkoussy/streamlit-fastapi-github-authentification


