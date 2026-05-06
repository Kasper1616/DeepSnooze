#!/bin/bash
# Inner worker called by knockout_eval.sh via bsub.
# Args: MODEL EXP_GROUP WANDB_GROUP NOTES
MODEL=$1
EXP_GROUP=$2
WANDB_GROUP=$3
NOTES=$4

SUBJECTS=(A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4 C5 C6 C7 C8 D1 D2 D3 D4 D5 D6)
N=${#SUBJECTS[@]}

for (( i=0; i<N; i++ )); do
    TEST=${SUBJECTS[i]}
    VAL=${SUBJECTS[$(( (i+1) % N ))]}
    EXP="${EXP_GROUP}_val_${VAL}_test_${TEST}"

    echo "--- Fold $((i+1))/${N}: ${EXP} ---"

    uv run python -m deepsnooze.evaluation.evaluate \
        model=${MODEL} \
        training=cv \
        data=cv \
        data.val_subject=${VAL} \
        data.test_subject=${TEST} \
        experiment_name=${EXP} \
        wandb.group=${WANDB_GROUP} \
        "wandb.notes='${NOTES}'"
done
