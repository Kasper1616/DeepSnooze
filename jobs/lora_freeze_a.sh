#!/bin/bash
# Cross-validation for LoRA and Bayesian LoRA with frozen A on cnn_standard, ranks 1/2, 100 epochs.
# Requires cnn_standard_cv fold checkpoints to exist in models/.
# Usage: bash jobs/lora_freeze_a.sh

NOTES="LoRA CV with frozen A matrix on frozen cnn_standard base. Comparing ranks 1, 2."
BASE_EXPERIMENT="cnn_standard_cv"
MODEL="cnn"

for RANK in 1 2; do

bsub << EOF
#!/bin/bash
#BSUB -J lora_cv_freeze_a_r\${RANK}
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_cv_freeze_a_r\${RANK}.out
#BSUB -e lora_cv_freeze_a_r\${RANK}.err

uv run python -m deepsnooze.train \
  model=\${MODEL} \
  data=cv \
  training=lora_freeze_a \
  training.rank=\${RANK} \
  training.max_epochs=100 \
  training.base_experiment=\${BASE_EXPERIMENT} \
  experiment_name=cnn_standard_lora_freeze_a_r\${RANK}_cv \
  wandb.group=lora_freeze_a_r\${RANK} \
  "wandb.notes='\${NOTES}'"
EOF

done

NOTES="Bayesian LoRA CV with frozen A matrix on frozen cnn_standard base. Comparing ranks 1, 2."
BASE_EXPERIMENT="cnn_standard_cv"
MODEL="cnn"
for RANK in 1 2; do

bsub << EOF
#!/bin/bash
#BSUB -J bayesian_lora_cv_freeze_a_r\${RANK}
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_cv_freeze_a_r\${RANK}.out
#BSUB -e bayesian_lora_cv_freeze_a_r\${RANK}.err

uv run python -m deepsnooze.train \
  model=\${MODEL} \
  data=cv \
  training=bayesian_lora_freeze_a \
  training.rank=\${RANK} \
  training.max_epochs=100 \
  training.base_experiment=\${BASE_EXPERIMENT} \
  experiment_name=cnn_standard_bayesian_lora_freeze_a_r\${RANK}_cv \
  wandb.group=bayesian_lora_freeze_a_r\${RANK} \
  "wandb.notes='\${NOTES}'"
EOF

done

echo "Submitted 4 CV jobs (lora_freeze_a + bayesian_lora_freeze_a, ranks 1 2)."
