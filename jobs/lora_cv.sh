#!/bin/bash
# Cross-validation for LoRA and Bayesian LoRA on cnn_standard, ranks 1/2/4/8, 100 epochs.
# Requires cnn_standard_cv fold checkpoints to exist in models/.
# Usage: bash jobs/lora_cv.sh

NOTES="LoRA CV on frozen cnn_standard base. Comparing ranks 1, 2, 4, 8."
BASE_EXPERIMENT="cnn_standard_cv"
MODEL="cnn"

for RANK in 1 2 4; do

bsub << EOF
#!/bin/bash
#BSUB -J lora_cv_r${RANK}
#BSUB -q gpuh100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_cv_r${RANK}.out
#BSUB -e lora_cv_r${RANK}.err

uv run python -m deepsnooze.train \
  model=${MODEL} \
  data=cv \
  training=lora \
  training.rank=${RANK} \
  training.max_epochs=100 \
  training.base_experiment=${BASE_EXPERIMENT} \
  experiment_name=cnn_standard_lora_r${RANK}_cv \
  wandb.group=lora_r${RANK} \
  "wandb.notes='${NOTES}'"
EOF

done

NOTES="Bayesian LoRA CV on frozen cnn_standard base. Comparing ranks 1, 2, 4, 8."
NOTES="LoRA CV on frozen cnn_standard base. Comparing ranks 1, 2, 4, 8."
BASE_EXPERIMENT="cnn_standard_cv"
MODEL="cnn"
for RANK in 1 2 4; do

bsub << EOF
#!/bin/bash
#BSUB -J bayesian_lora_cv_r${RANK}
#BSUB -q gpuh100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00

uv run python -m deepsnooze.train \
  model=${MODEL} \
  data=cv \
  training=bayesian_lora \
  training.rank=${RANK} \
  training.max_epochs=100 \
  training.base_experiment=${BASE_EXPERIMENT} \
  experiment_name=cnn_standard_bayesian_lora_r${RANK}_cv \
  wandb.group=bayesian_lora_r${RANK} \
  "wandb.notes='${NOTES}'"
EOF

done

echo "Submitted 8 CV jobs (lora + bayesian_lora, ranks 1 2 4 8)."
