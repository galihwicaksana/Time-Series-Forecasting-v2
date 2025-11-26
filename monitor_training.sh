#!/bin/bash

# Script to monitor LSTM training progress

LOG_DIR="logs/lstm_etth1"
LATEST_LOG=$(ls -t ${LOG_DIR}/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training logs found in ${LOG_DIR}/"
    exit 1
fi

echo "========================================"
echo "LSTM Training Monitor"
echo "========================================"
echo "Log file: ${LATEST_LOG}"
echo ""
echo "Commands:"
echo "  - Attach to tmux: tmux attach -t lstm_training"
echo "  - Kill training: tmux kill-session -t lstm_training"
echo "  - View all logs: cat ${LATEST_LOG}"
echo "========================================"
echo ""

# Check if tmux session is running
if tmux has-session -t lstm_training 2>/dev/null; then
    echo "Status: Training is RUNNING in tmux"
else
    echo "Status: No active tmux session (training may be completed or stopped)"
fi

echo ""
echo "Latest log output:"
echo "----------------------------------------"
tail -50 "$LATEST_LOG"
echo "----------------------------------------"
echo ""
echo "To follow logs in real-time: tail -f ${LATEST_LOG}"
