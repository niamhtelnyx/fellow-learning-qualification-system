#!/bin/bash
# Start Fellow Learning API Server

cd /Users/niamhcollins/clawd/fellow-learning-system
export PYTHONPATH=$PYTHONPATH:/Users/niamhcollins/clawd/fellow-learning-system/ml-model:/Users/niamhcollins/clawd/fellow-learning-system/api

echo "Starting Fellow Learning API..."
python -m uvicorn api.lead_scorer:app --host 0.0.0.0 --port 8000 --reload
