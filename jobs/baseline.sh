#!/bin/bash
# Submit baseline runs (standard CE loss) for all 3 CNN architectures.
# Usage: bash jobs/baseline.sh

GROUP="baseline"
NOTES="Standard cross-entropy loss. Comparing SimpleCNN, SleepyCNN, and DeepCNN architectures."

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_simple
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_simple_%J.out
#BSUB -e ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_simple_%J.err

uv run python -m deepsnooze.train model=cnn_simple training=default wandb.group=${GROUP} "wandb.notes=${NOTES}"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_medium
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_medium_%J.out
#BSUB -e ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_medium_%J.err

uv run python -m deepsnooze.train model=cnn training=default wandb.group=${GROUP} "wandb.notes=${NOTES}"
EOF

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_deep
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_deep_%J.out
#BSUB -e ~/Desktop/Deepsnooze/logs/lsf/${GROUP}_deep_%J.err

uv run python -m deepsnooze.train model=cnn_deep training=default wandb.group=${GROUP} "wandb.notes=${NOTES}"
EOF

echo "Submitted 3 baseline jobs."
