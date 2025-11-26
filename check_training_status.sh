#!/bin/bash

# Script to check training status and summarize results

echo "========================================"
echo "LSTM Training Status & Results"
echo "========================================"
echo ""

# Check tmux session
if tmux has-session -t lstm_training 2>/dev/null; then
    echo "✓ Training is RUNNING in tmux session"
    echo "  Attach with: tmux attach -t lstm_training"
else
    echo "✗ No active tmux session"
    echo "  Training may be completed or stopped"
fi

echo ""
echo "========================================"
echo "Log Files"
echo "========================================"
ls -lth logs/lstm_etth1/training_*.log 2>/dev/null | head -5 || echo "No log files found"

echo ""
echo "========================================"
echo "Results"
echo "========================================"

# Check for results
if [ -f "results/lstm_etth1/summary_results.json" ]; then
    echo "✓ Summary results found:"
    cat results/lstm_etth1/summary_results.json | python3 -m json.tool 2>/dev/null || cat results/lstm_etth1/summary_results.json
else
    echo "✗ No summary results yet"
fi

echo ""

if [ -f "results/lstm_etth1/lstm_etth1_results.csv" ]; then
    echo "✓ CSV results found:"
    cat results/lstm_etth1/lstm_etth1_results.csv
else
    echo "✗ No CSV results yet"
fi

echo ""
echo "========================================"
echo "Saved Models"
echo "========================================"
for dir in checkpoints/lstm_etth1_*; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
        ls -lh "$dir"/*.h5 2>/dev/null | wc -l | xargs echo "  Models:"
    fi
done

echo ""
echo "========================================"
echo "Latest Log Excerpt (last 30 lines):"
echo "========================================"
LATEST_LOG=$(ls -t logs/lstm_etth1/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -30 "$LATEST_LOG"
else
    echo "No logs available"
fi
