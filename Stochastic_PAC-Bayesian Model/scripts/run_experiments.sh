#!/bin/bash
# =============================================================================
# Run full experimental pipeline across all datasets and folds.
#
# Reproduces Tables 1-5 and Figures 2-3 from the paper.
#
# Prerequisites:
#   1. Download the Integrated IDPS Security 3Datasets:
#      https://doi.org/10.34740/KAGGLE/DSV/12479689
#   2. Download fake news datasets from their respective repositories
#   3. Install requirements: pip install -r requirements.txt
#
# Usage:
#   bash scripts/run_experiments.sh /path/to/data /path/to/output
# =============================================================================

set -euo pipefail

DATA_DIR="${1:-data}"
OUTPUT_DIR="${2:-outputs}"
CONFIG="configs/default_config.yaml"
SEEDS=(42 123 456)

echo "============================================="
echo "Stochastic PAC-Bayesian Transformer Experiments"
echo "============================================="
echo "Data directory:   $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# --- Network Intrusion Detection ---
for dataset in cic_iot_2023 cse_cicids2018 unsw_nb15; do
    for fold in 0 1 2 3 4; do
        for seed in "${SEEDS[@]}"; do
            echo "[TRAIN] $dataset fold=$fold seed=$seed"
            python scripts/train.py \
                --domain network \
                --dataset "$dataset" \
                --data_path "$DATA_DIR/$dataset.csv" \
                --config "$CONFIG" \
                --fold "$fold" \
                --seed "$seed" \
                --output_dir "$OUTPUT_DIR/seed_$seed"

            echo "[EVAL] $dataset fold=$fold seed=$seed"
            python scripts/evaluate.py \
                --checkpoint "$OUTPUT_DIR/seed_$seed/$dataset/fold_$fold/best_model.pt" \
                --domain network \
                --dataset "$dataset" \
                --data_path "$DATA_DIR/$dataset.csv" \
                --config "$CONFIG" \
                --fold "$fold" \
                --output_dir "$OUTPUT_DIR/results/seed_$seed"
        done
    done
done

# --- Toxic Content Detection ---
for dataset in metahate hateval founta; do
    for fold in 0 1 2 3 4; do
        for seed in "${SEEDS[@]}"; do
            echo "[TRAIN] $dataset fold=$fold seed=$seed"
            python scripts/train.py \
                --domain toxic \
                --dataset "$dataset" \
                --data_path "$DATA_DIR/$dataset.csv" \
                --config "$CONFIG" \
                --fold "$fold" \
                --seed "$seed" \
                --output_dir "$OUTPUT_DIR/seed_$seed"
        done
    done
done

# --- Fake News Detection ---
for dataset in liar fakenewsnet isot; do
    for fold in 0 1 2 3 4; do
        for seed in "${SEEDS[@]}"; do
            echo "[TRAIN] $dataset fold=$fold seed=$seed"
            python scripts/train.py \
                --domain fakenews \
                --dataset "$dataset" \
                --data_path "$DATA_DIR/$dataset.csv" \
                --config "$CONFIG" \
                --fold "$fold" \
                --seed "$seed" \
                --output_dir "$OUTPUT_DIR/seed_$seed"
        done
    done
done

echo ""
echo "============================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR/results/"
echo "============================================="
