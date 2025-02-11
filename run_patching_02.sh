#!/bin/bash
while true; do
    python preprocessing/main_02_patching_temp50.py
    echo "Process was killed. Restarting in 5 seconds..."
    sleep 5
done