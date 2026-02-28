#!/bin/bash

# =================================================
# EMG UMAP Test Configuration Runner
# =================================================
# This script allows you to run the umap_test.py with different configurations

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
    echo "Usage: ./run_umap.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  default              Run with default configuration (Ricardo's Open/Close Hand)"
    echo "  ricardo              Run with Ricardo's mixed test data"
    echo "  all                  Run with all subjects (Ricardo + Sirisha)"
    echo "  emc                  Run with EMG_ASL folder test data"
    echo "  random               Run with Random Forest dataset"
    echo "  fingers              Run with individual finger tests (Index, Ring, Pinky)"
    echo "  hand                 Run with hand position variants (Closed Hand + Fingers)"
    echo "  all_gestures         Run with complete gesture set"
    echo "  comparison           Run with subject comparison (Ricardo vs Mohak - Closed Hand)"
    echo "  custom <file> <label1> <label2> ... <fileN> [<labelN>]"
    echo "                       Run with custom files and labels"
    echo "  list                 List all available configurations"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_umap.sh default"
    echo "  ./run_umap.sh fingers"
    echo "  ./run_umap.sh comparison"
    echo "  ./run_umap.sh custom CSV-Files/Test-Ricardo.csv Ricardo CSV-Files/index_finger.csv 'Index Finger'"
}

# Function to run with config file
run_with_config() {
    local config_key=$1
    echo -e "${YELLOW}Running UMAP test with configuration: $config_key${NC}"
    python umap_test.py --config gesture_configs.json
}

# Function to run with command-line arguments
run_with_args() {
    local files=()
    local labels=()
    
    # Parse arguments (alternating file and label)
    while [[ $# -gt 0 ]]; do
        files+=("$1")
        shift
        if [[ $# -gt 0 ]]; then
            labels+=("$1")
            shift
        fi
    done
    
    # Check that we have matching files and labels
    if [[ ${#files[@]} -ne ${#labels[@]} ]]; then
        echo -e "${RED}Error: Number of files must match number of labels${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Running UMAP test with custom configuration${NC}"
    echo -e "${GREEN}Files: ${files[@]}${NC}"
    echo -e "${GREEN}Labels: ${labels[@]}${NC}"
    python umap_test.py --files "${files[@]}" --labels "${labels[@]}"
}

# Main logic
if [[ $# -eq 0 ]]; then
    show_usage
    exit 0
fi

case "$1" in
    default)
        echo -e "${GREEN}✓ Using default configuration${NC}"
        python umap_test.py
        ;;
    ricardo)
        echo -e "${GREEN}✓ Using Ricardo only configuration${NC}"
        python umap_test.py --files "CSV-Files/Test-Ricardo.csv" --labels "Ricardo Mixed"
        ;;
    all)
        echo -e "${GREEN}✓ Using all subjects configuration${NC}"
        python umap_test.py --files "CSV-Files/Test-Ricardo.csv" "CSV-Files/Test-Sirisha.csv" --labels "Ricardo" "Sirisha"
        ;;
    emc)
        echo -e "${GREEN}✓ Using EMG_ASL configuration${NC}"
        python umap_test.py --files "EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv" "EMG_ASL/CSV/Test-Ricardo_Closed-Hand.csv" --labels "Open Hand" "Close Hand"
        ;;
    random)
        echo -e "${GREEN}✓ Using Random Forest dataset configuration${NC}"
        python umap_test.py --files "RandomForest/DATASET EMG MINDROVE/AYU/subjek_AYU1.csv" "RandomForest/DATASET EMG MINDROVE/DANIEL/subjek_DANIEL1.csv" "RandomForest/DATASET EMG MINDROVE/LINTANG/subjek_Lintang1.csv" --labels "AYU-1" "DANIEL-1" "LINTANG-1"
        ;;
    fingers)
        echo -e "${GREEN}✓ Using finger tests configuration${NC}"
        python umap_test.py --files "CSV-Files/index_finger.csv" "CSV-Files/ring_finger.csv" "CSV-Files/pinky_finger.csv" --labels "Index Finger" "Ring Finger" "Pinky Finger"
        ;;
    hand)
        echo -e "${GREEN}✓ Using hand position variants configuration${NC}"
        python umap_test.py --files "CSV-Files/closed_hand.csv" "CSV-Files/index_finger.csv" "CSV-Files/ring_finger.csv" "CSV-Files/pinky_finger.csv" --labels "Closed Hand" "Index Finger" "Ring Finger" "Pinky Finger"
        ;;
    all_gestures)
        echo -e "${GREEN}✓ Using complete gesture set configuration${NC}"
        python umap_test.py --files "CSV-Files/Test-Ricardo_Open-Hand.csv" "CSV-Files/closed_hand.csv" "CSV-Files/index_finger.csv" "CSV-Files/ring_finger.csv" "CSV-Files/pinky_finger.csv" --labels "Open Hand" "Closed Hand" "Index Finger" "Ring Finger" "Pinky Finger"
        ;;
    comparison)
        echo -e "${GREEN}✓ Using subject comparison configuration (Ricardo vs Mohak)${NC}"
        python umap_test.py --files "CSV-Files/closed_hand.csv" "CSV-Files/Test-Mohak_Closed-Hand.csv" --labels "Ricardo - Closed Hand" "Mohak - Closed Hand"
        ;;
    custom)
        shift  # Remove 'custom' argument
        run_with_args "$@"
        ;;
    list)
        echo -e "${GREEN}Available configurations in gesture_configs.json:${NC}"
        python -c "
import json
with open('gesture_configs.json', 'r') as f:
    configs = json.load(f)
    for key, config in configs.items():
        print(f'\n{key}:')
        print(f'  Name: {config[\"name\"]}')
        print(f'  Description: {config[\"description\"]}')
        print(f'  Gestures:')
        for gesture in config['gestures']:
            print(f'    - {gesture[\"label\"]}: {gesture[\"file\"]}')
"
        ;;
    help)
        show_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown option '$1'${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Script completed successfully${NC}"
