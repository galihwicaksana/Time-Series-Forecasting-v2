#!/bin/bash

# Script to run LSTM training in tmux with logging
# This script will run training in background and save all logs

# Configuration
SESSION_NAME="lstm_training"
LOG_DIR="logs/lstm_etth1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

# Check if tmux session already exists
tmux has-session -t ${SESSION_NAME} 2>/dev/null

if [ $? == 0 ]; then
    echo "Tmux session '${SESSION_NAME}' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill and restart: tmux kill-session -t ${SESSION_NAME} && $0"
    exit 1
fi

# Create new tmux session and run training
echo "========================================"
echo "Starting LSTM Training in tmux"
echo "========================================"
echo "Session name: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Commands:"
echo "  - Attach to session: tmux attach -t ${SESSION_NAME}"
echo "  - Detach from session: Ctrl+b then d"
echo "  - View logs: tail -f ${LOG_FILE}"
echo "  - Kill session: tmux kill-session -t ${SESSION_NAME}"
echo "========================================"
echo ""

# Create tmux session and run training
tmux new-session -d -s ${SESSION_NAME} -n "LSTM_Training" \
    "cd /mnt/extended-home/galih/Time-Series-Library && \
     source venv/bin/activate && \
     echo 'LSTM Training Started at: $(date)' | tee ${LOG_FILE} && \
     echo 'Log file: ${LOG_FILE}' | tee -a ${LOG_FILE} && \
     echo '========================================' | tee -a ${LOG_FILE} && \
     python run_lstm_etth1_comparable.py 2>&1 | tee -a ${LOG_FILE} ; \
     echo '' | tee -a ${LOG_FILE} && \
     echo '========================================' | tee -a ${LOG_FILE} && \
     echo 'Training completed at: $(date)' | tee -a ${LOG_FILE} && \
     echo 'Press Enter to close this window or Ctrl+b d to detach' && \
     read"

echo "Training started in background!"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "Or view logs in real-time:"
echo "  tail -f ${LOG_FILE}"
echo ""
