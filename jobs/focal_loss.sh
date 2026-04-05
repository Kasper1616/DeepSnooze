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
#BSUB -o ${GROUP}_simple.out
#BSUB -e ${GROUP}_simple.err

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
#BSUB -o ${GROUP}_medium.out
#BSUB -e ${GROUP}_medium.err

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
#BSUB -o ${GROUP}_deep.out
#BSUB -e ${GROUP}_deep.err

uv run python -m deepsnooze.train model=cnn_deep training=focal_loss wandb.group=${GROUP} "wandb.notes=${NOTES}"
EOF

echo "Submitted 3 focal loss jobs."
