#!/bin/bash
# Submit LoRA and Bayesian LoRA runs on cnn_deep, comparing rank 1 and rank 5.
# Requires cnn_deep_base.pt to exist in models/ (saved after baseline training).
# Usage: bash jobs/lora.sh

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_lora_deep_r1
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_deep_r1.out
#BSUB -e lora_deep_r1.err

source ~/.bashrc
cd ~/Desktop/Deepsnooze
uv run python -m deepsnooze.train model=cnn_deep training=lora training.rank=1 wandb.group=lora "wandb.notes='Standard LoRA rank=1 on frozen cnn_deep base.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_lora_deep_r5
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_deep_r5.out
#BSUB -e lora_deep_r5.err

source ~/.bashrc
cd ~/Desktop/Deepsnooze
uv run python -m deepsnooze.train model=cnn_deep training=lora training.rank=5 wandb.group=lora "wandb.notes='Standard LoRA rank=5 on frozen cnn_deep base.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_bayesian_lora_deep_r1
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_deep_r1.out
#BSUB -e bayesian_lora_deep_r1.err

source ~/.bashrc
cd ~/Desktop/Deepsnooze
uv run python -m deepsnooze.train model=cnn_deep training=bayesian_lora training.rank=1 wandb.group=lora "wandb.notes='Bayesian LoRA rank=1 on frozen cnn_deep base. Uncertainty-aware FC layers via Pyro SVI.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_bayesian_lora_deep_r5
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_deep_r5.out
#BSUB -e bayesian_lora_deep_r5.err

source ~/.bashrc
cd ~/Desktop/Deepsnooze
uv run python -m deepsnooze.train model=cnn_deep training=bayesian_lora training.rank=5 wandb.group=lora "wandb.notes='Bayesian LoRA rank=5 on frozen cnn_deep base. Uncertainty-aware FC layers via Pyro SVI.'"
EOF

echo "Submitted 4 LoRA jobs (standard r1, standard r5, bayesian r1, bayesian r5)."
