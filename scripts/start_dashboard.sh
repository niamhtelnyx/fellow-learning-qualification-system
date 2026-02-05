#!/bin/bash
# Start Fellow Learning Dashboard

cd /Users/niamhcollins/clawd/fellow-learning-system
export PYTHONPATH=$PYTHONPATH:/Users/niamhcollins/clawd/fellow-learning-system/ml-model:/Users/niamhcollins/clawd/fellow-learning-system/api

echo "Starting Fellow Learning Dashboard..."
streamlit run dashboard/performance_dashboard.py --server.port 8501
