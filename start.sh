#!/bin/bash

echo "Starting Python application..."
echo "Binding to 0.0.0.0:7860"
python -u gradio_mcp.py --host 0.0.0.0 --port 7860 --graph-file /app/pedagogia_graph_code_repo/data/hf-repos/transformers/multihop_knowledge_graph_with_embeddings.json
