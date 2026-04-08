#!/bin/bash
# Submit Bayesian LoRA runs with full multivariate normal variational family.
# Compares against mean-field (AutoNormal) from lora.sh.
# Requires cnn_deep_base.pt to exist in models/ (saved after baseline training).
# Usage: bash jobs/lora_mvn.sh

for RANK in 1 3 5 8; do

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_bayesian_lora_mvn_r${RANK}
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_mvn_r${RANK}.out
#BSUB -e bayesian_lora_mvn_r${RANK}.err

source ~/.bashrc
cd ~/Desktop/Deepsnooze
uv run python -m deepsnooze.train model=cnn_deep training=bayesian_lora_mvn training.rank=${RANK} wandb.group=lora_mvn "wandb.notes='Bayesian LoRA rank=${RANK} with full multivariate normal posterior. Captures correlations between A and B matrices.'"
EOF

done

echo "Submitted 4 Bayesian LoRA MVN jobs (ranks 1 3 5 8)."
