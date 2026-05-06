#!/bin/bash
# Submit knockout robustness evaluation for all 3 CNNs x 2 training modes (cv, knockout).
# Each job runs all 22 CV folds via jobs/_eval_folds.sh.
# Usage: bash jobs/knockout_eval.sh

NOTES="Knockout robustness evaluation — clean vs ko_ch0/1/2 at test time."

submit() {
    local JOB_NAME=$1 MODEL=$2 EXP_GROUP=$3 WANDB_GROUP=$4

    bsub -J  "${JOB_NAME}" \
         -q  gpua40 \
         -gpu "num=1:mode=exclusive_process" \
         -n  8 \
         -R  "rusage[mem=8GB]" \
         -W  04:00 \
         -o  "logs/${JOB_NAME}.out" \
         -e  "logs/${JOB_NAME}.err" \
         bash jobs/_eval_folds.sh "${MODEL}" "${EXP_GROUP}" "${WANDB_GROUP}" "${NOTES}"
}

# --- Baseline (trained without knockout) ---
submit ko_eval_simple_cv    cnn_simple  cnn_simple_cv    knockout_eval_cv
submit ko_eval_standard_cv  cnn         cnn_standard_cv  knockout_eval_cv
submit ko_eval_deep_cv      cnn_deep    cnn_deep_cv      knockout_eval_cv

# --- Knockout-trained ---
submit ko_eval_simple_ko    cnn_simple  cnn_simple_knockout    knockout_eval_ko
submit ko_eval_standard_ko  cnn         cnn_standard_knockout  knockout_eval_ko
submit ko_eval_deep_ko      cnn_deep    cnn_deep_knockout      knockout_eval_ko

echo "Submitted 6 evaluation jobs (3 models x 2 training modes, 22 folds each)."
