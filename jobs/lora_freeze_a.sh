#!/bin/bash
# Submit LoRA and Bayesian LoRA runs with frozen A matrix on cnn_deep, testing rank 1 and 2.
# Usage: bash jobs/lora_freeze_a.sh

bsub << EOF
#!/bin/bash
#BSUB -J lora_deep_r1_freeze_a
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_deep_r1_freeze_a.out
#BSUB -e lora_deep_r1_freeze_a.err

uv run python -m deepsnooze.train model=cnn_deep training=lora_freeze_a training.rank=1 wandb.group=lora_freeze_a "wandb.notes='LoRA rank=1 with frozen A matrix.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J bayesian_lora_deep_r1_freeze_a
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_deep_r1_freeze_a.out
#BSUB -e bayesian_lora_deep_r1_freeze_a.err

uv run python -m deepsnooze.train model=cnn_deep training=bayesian_lora_freeze_a training.rank=1 wandb.group=lora_freeze_a "wandb.notes='Bayesian LoRA rank=1 with frozen A matrix.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J lora_deep_r2_freeze_a
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o lora_deep_r2_freeze_a.out
#BSUB -e lora_deep_r2_freeze_a.err

uv run python -m deepsnooze.train model=cnn_deep training=lora_freeze_a training.rank=2 wandb.group=lora_freeze_a "wandb.notes='LoRA rank=2 with frozen A matrix.'"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J bayesian_lora_deep_r2_freeze_a
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o bayesian_lora_deep_r2_freeze_a.out
#BSUB -e bayesian_lora_deep_r2_freeze_a.err

uv run python -m deepsnooze.train model=cnn_deep training=bayesian_lora_freeze_a training.rank=2 wandb.group=lora_freeze_a "wandb.notes='Bayesian LoRA rank=2 with frozen A matrix.'"
EOF

echo "Submitted 4 LoRA freeze_A jobs (standard + bayesian, ranks 1, 2)."
