#!/bin/bash
INPUT_FILE="./datasets/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad"
OUTPUT_DIR="./results/minimal_imputation"
FIGURES_DIR="./results/figures"
CACHE_DIR="./results/cache"
MARKERS_FILE="./data/marker_genes.json"
mkdir -p "$OUTPUT_DIR" "$FIGURES_DIR" "$CACHE_DIR"

# Assuming the Python file is named minimal_imputation.py
python th.py \
--input_file "$INPUT_FILE" \
--output_dir "$OUTPUT_DIR" \
--figures_dir "$FIGURES_DIR" \
--cache_dir "$CACHE_DIR" \
--markers_file "$MARKERS_FILE" \
--methods "SAUCIE,scVI,MAGIC,deepImpute" \
--batch_percent 1.0 \
--early_stop 10

# Generate example marker genes JSON if not provided
if [ ! -f "$MARKERS_FILE" ]; then
echo "Creating example marker genes file at $MARKERS_FILE"
mkdir -p $(dirname "$MARKERS_FILE")
cat > "$MARKERS_FILE" << EOF
{
 "Enterocytes BEST4": [
 "BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B", "SPIB", "CFTR"
 ],
 "Goblet cells MUC2 TFF1": [
 "MUC2", "TFF1", "TFF3", "FCGBP", "AGR2", "SPDEF"
 ],
 "Tuft cells": [
 "POU2F3", "DCLK1"
 ],
 "Goblet cells SPINK4": [
 "MUC2", "SPINK4"
 ],
 "Enterocytes TMIGD1 MEP1A": [
 "CA1", "CA2", "TMIGD1", "MEP1A"
 ],
 "Enterocytes CA1 CA2 CA4-": [
 "CA1", "CA2"
 ],
 "Goblet cells MUC2 TFF1-": [
 "MUC2"
 ],
 "Epithelial Cycling cells": [
 "LGR5", "OLFM4", "MKI67"
 ],
 "Enteroendocrine cells": [
 "CHGA", "GCG", "GIP", "CCK"
 ],
 "Stem cells OLFM4": [
 "OLFM4", "LGR5"
 ],
 "Stem cells OLFM4 LGR5": [
 "OLFM4", "LGR5", "ASCL2"
 ],
 "Stem cells OLFM4 PCNA": [
 "OLFM4", "PCNA", "LGR5", "ASCL2", "SOX9", "TERT"
 ],
 "Paneth cells": [
 "LYZ", "DEFA5"
 ]
}
EOF
fi

# Print completion message
echo "Experiment completed. Results saved to $OUTPUT_DIR"
echo "Figures saved to $FIGURES_DIR"
echo "Reports available at:"
echo "  - $OUTPUT_DIR/minimal_imputation_results.csv"
echo "  - $OUTPUT_DIR/minimal_imputation_summary.txt"
echo "  - $OUTPUT_DIR/minimal_imputation_report.txt"



Below are **concise mathematical expressions** for the trustworthiness (TH) score and the consensus score. In each formula, we consider $M$ imputation methods and let $\hat{X}_m(c,g)$ be the imputed value for cell $c$, gene $g$ under method $m$. The **indicator function** $\mathbf{1}\{\cdot\}$ is 1 if its argument is true and 0 otherwise.

---

**Trustworthiness (TH) Score**
Measures the *fraction* of methods that impute a nonzero value:

\[
\text{TH}(c,g) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\!\bigl(\,\hat{X}_m(c,g)>0\bigr).
\]

\[

\]

**Consensus Score**
Takes the *average* of each method'’'s imputed values:

\[
\text{Consensus}(c,g) = \frac{1}{M}\sum_{m=1}^{M} \hat{X}_m(c,g).
\]
