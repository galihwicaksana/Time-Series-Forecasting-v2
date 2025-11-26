#!/bin/bash

################################################################################
# Master Script untuk Menjalankan Semua Eksperimen
# Dataset: ETTh1 & ETTh2
# Models: TimeMixer, TimesNet, Transformer
# Sequence Lengths: [48, 96, 168, 336]
# Prediction Lengths: [96, 192, 336, 720]
# Total: 6 scripts × 16 experiments = 96 training runs
################################################################################

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file master
MASTER_LOG="master_training_log.txt"

# Fungsi untuk print dengan timestamp
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${timestamp}]${NC} ${message}"
    echo "[${timestamp}] ${message}" >> $MASTER_LOG
}

# Fungsi untuk print header
print_header() {
    local text="$1"
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}${text}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Initialize master log
echo "Master Training Log" > $MASTER_LOG
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" >> $MASTER_LOG
echo "========================================" >> $MASTER_LOG
echo "" >> $MASTER_LOG

# Simpan waktu mulai
START_TIME=$(date +%s)

print_header "🚀 Starting Comprehensive Training Experiment"
log_message "Total Experiments: 96 training runs (6 scripts × 16 combinations)"
log_message "Estimated Total Time: 10-14 hours"
echo ""

# Array of scripts
declare -a SCRIPTS=(
    "scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh"
    "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh"
    "scripts/long_term_forecast/ETT_script/Transformer_ETTh1.sh"
    "scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2.sh"
    "scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh"
    "scripts/long_term_forecast/ETT_script/Transformer_ETTh2.sh"
)

declare -a SCRIPT_NAMES=(
    "TimeMixer (ETTh1)"
    "TimesNet (ETTh1)"
    "Transformer (ETTh1)"
    "TimeMixer (ETTh2)"
    "TimesNet (ETTh2)"
    "Transformer (ETTh2)"
)

# Counter untuk progress
TOTAL_SCRIPTS=${#SCRIPTS[@]}
CURRENT_SCRIPT=0

# Loop untuk setiap script
for i in "${!SCRIPTS[@]}"; do
    SCRIPT="${SCRIPTS[$i]}"
    NAME="${SCRIPT_NAMES[$i]}"
    CURRENT_SCRIPT=$((CURRENT_SCRIPT + 1))
    
    print_header "📊 Script ${CURRENT_SCRIPT}/${TOTAL_SCRIPTS}: ${NAME}"
    log_message "Starting: ${NAME}"
    
    SCRIPT_START=$(date +%s)
    
    # Jalankan script
    if bash "$SCRIPT"; then
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        SCRIPT_MINUTES=$((SCRIPT_DURATION / 60))
        SCRIPT_SECONDS=$((SCRIPT_DURATION % 60))
        
        echo -e "${GREEN}✓ Completed: ${NAME}${NC}"
        log_message "✓ Completed: ${NAME} (Duration: ${SCRIPT_MINUTES}m ${SCRIPT_SECONDS}s)"
    else
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        SCRIPT_MINUTES=$((SCRIPT_DURATION / 60))
        SCRIPT_SECONDS=$((SCRIPT_DURATION % 60))
        
        echo -e "${RED}✗ Failed: ${NAME}${NC}"
        log_message "✗ Failed: ${NAME} (Duration: ${SCRIPT_MINUTES}m ${SCRIPT_SECONDS}s)"
        
        # Tanyakan apakah ingin melanjutkan
        read -p "Continue with remaining scripts? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_message "Training stopped by user after failure"
            exit 1
        fi
    fi
    
    # Progress indicator
    ELAPSED=$(($(date +%s) - START_TIME))
    ELAPSED_MINUTES=$((ELAPSED / 60))
    ELAPSED_HOURS=$((ELAPSED_MINUTES / 60))
    ELAPSED_MINUTES_REMAIN=$((ELAPSED_MINUTES % 60))
    
    echo ""
    echo -e "${YELLOW}Progress: ${CURRENT_SCRIPT}/${TOTAL_SCRIPTS} completed${NC}"
    echo -e "${YELLOW}Elapsed Time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES_REMAIN}m${NC}"
    echo ""
    
    # Jeda singkat antar script
    sleep 2
done

# Hitung total waktu
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

print_header "🎉 All Training Completed!"
log_message "All experiments completed successfully!"
log_message "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"

echo ""
echo -e "${GREEN}Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s${NC}"
echo ""
echo -e "${BLUE}Log files generated:${NC}"
echo "  - master_training_log.txt"
echo "  - long_term_forecast_timeMixer_ETTh1_results.log"
echo "  - long_term_forecast_timesNet_ETTh1_results.log"
echo "  - long_term_forecast_transformer_ETTh1_results.log"
echo "  - long_term_forecast_timeMixer_ETTh2_results.log"
echo "  - long_term_forecast_timesNet_ETTh2_results.log"
echo "  - long_term_forecast_transformer_ETTh2_results.log"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Run visualizations:"
echo "     python visualize_results.py --dataset ETTh1"
echo "     python visualize_results.py --dataset ETTh2"
echo ""
echo "  2. Check results in:"
echo "     - checkpoints/ (model weights)"
echo "     - results/ (predictions)"
echo "     - visualizations/ (plots after running visualize_results.py)"
echo ""

# Final summary
echo "" >> $MASTER_LOG
echo "========================================" >> $MASTER_LOG
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> $MASTER_LOG
echo "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" >> $MASTER_LOG
echo "========================================" >> $MASTER_LOG
