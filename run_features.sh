#!/bin/bash
while true; do
    python preprocessing/main_03_get_feature_draft.py
    echo "Process was killed. Restarting in 5 seconds..."
    sleep 5
done