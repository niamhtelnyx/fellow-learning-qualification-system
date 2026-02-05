#!/bin/bash
# Run continuous learning cycle

cd /Users/niamhcollins/clawd/fellow-learning-system
export PYTHONPATH=$PYTHONPATH:/Users/niamhcollins/clawd/fellow-learning-system/ml-model

echo "Running continuous learning cycle..."
python ml-model/continuous_learner.py
