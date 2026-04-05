#!/bin/bash
# Submit focal loss runs for all 3 CNN architectures.
# Usage: bash jobs/focal_loss.sh

GROUP="focal_loss"
NOTES="Focal loss (gamma=2.0) with class weights. Targets hard examples, expected to improve REM recall."

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

cd ~/Desktop/Deepsnooze
source .venv/bin/activate
uv run python -m deepsnooze.train model=cnn_simple training=focal_loss wandb.group=${GROUP} "wandb.notes=${NOTES}"
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

cd ~/Desktop/Deepsnooze
source .venv/bin/activate
uv run python -m deepsnooze.train model=cnn training=focal_loss wandb.group=${GROUP} "wandb.notes=${NOTES}"
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

cd ~/Desktop/Deepsnooze
source .venv/bin/activate
uv run python -m deepsnooze.train model=cnn_deep training=focal_loss wandb.group=${GROUP} "wandb.notes=${NOTES}"
EOF

echo "Submitted 3 focal loss jobs."
