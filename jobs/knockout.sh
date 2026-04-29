
NOTES="Knockout Test."
GROUP="cnn_simple_knockout"


bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_simple
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o ${GROUP}_simple.out
#BSUB -e ${GROUP}_simple.err

source ~/.bashrc

cd /zhome/96/e/167682/rml/DeepSnooze # change this

uv run python -m deepsnooze.train model=cnn_simple data=knockout experiment_name=${GROUP} "wandb.notes='${NOTES}'"
EOF


GROUP="cnn_standard_knockout"


bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_standard
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o ${GROUP}_standard.out
#BSUB -e ${GROUP}_standard.err

source ~/.bashrc

cd /zhome/96/e/167682/rml/DeepSnooze

uv run python -m deepsnooze.train model=cnn data=knockout experiment_name=${GROUP} "wandb.notes='${NOTES}'"
EOF

GROUP="cnn_deep_knockout"


bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_deep
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o ${GROUP}_deep.out
#BSUB -e ${GROUP}_deep.err

source ~/.bashrc

cd /zhome/96/e/167682/rml/DeepSnooze

uv run python -m deepsnooze.train model=cnn_deep data=knockout experiment_name=${GROUP} "wandb.notes='${NOTES}'"
EOF