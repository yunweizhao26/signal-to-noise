#!/usr/bin/env bash
#################### 用户可修改区域 ####################
INPUT_FILE="./datasets/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad"
OUTPUT_DIR="./results/minimal_imputation"
FIGURES_DIR="./results/figures"
CACHE_DIR="./results/cache"
MARKERS_FILE="./data/marker_genes.json"
METHODS="SAUCIE,MAGIC,deepImpute,scScope,scVI,knn_smoothing" # 若有其它方法，可在此逗号分隔添加
BATCH_PERCENT=0 # 0=自适应；固定 1% 就填 1.0
EARLY_STOP=10 # 连续多少步无显著提升后提前停止
BOOTSTRAPS=20 # 不需要不确定度分析可设 0
IMPROVE_THRES=0.001 # Δscore 小于此阈值记为"无提升"
########################################################
# 1. 创建结果/缓存/marker 目录
mkdir -p "$OUTPUT_DIR" "$FIGURES_DIR" "$CACHE_DIR" "$(dirname "$MARKERS_FILE")"
# 3. 运行 Python 实验
python th_3.py \
--input_file "$INPUT_FILE" \
--output_dir "$OUTPUT_DIR" \
--figures_dir "$FIGURES_DIR" \
--cache_dir "$CACHE_DIR" \
--markers_file "$MARKERS_FILE" \
--methods "$METHODS" \
--batch_percent "$BATCH_PERCENT" \
--early_stop "$EARLY_STOP" \
--bootstraps "$BOOTSTRAPS" \
--improvement_threshold "$IMPROVE_THRES" \
--verbose
# 4. 完成提示
echo -e "\n 实验已完成"
echo "结果目录: $OUTPUT_DIR"
echo "图形目录: $FIGURES_DIR"
echo "主要报告:"
echo " - $OUTPUT_DIR/minimal_imputation_results.csv"
echo " - $OUTPUT_DIR/minimal_imputation_summary.txt"
echo " - $OUTPUT_DIR/optimal_percentage_statistics.txt"