#!/bin/bash
# Start the batch evaluation with docker compose

# Create eval_results directory if it doesn't exist
mkdir -p eval_results

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the services
docker compose -f docker-compose-batch-eval.yml up --build

echo "Batch evaluation complete. Results saved in ./eval_results/"
