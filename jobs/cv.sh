
NOTES="Standard cross-entropy loss. Comparing SimpleCNN, SleepyCNN, and DeepCNN architectures."
GROUP="cnn_simple_cv"


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

cd /zhome/96/e/167682/rml/DeepSnooze

uv run python -m deepsnooze.train model=cnn_simple data=cv training=cv experiment_name=${GROUP} "wandb.notes='${NOTES}'"
EOF

GROUP="cnn_standard_cv"

bsub << EOF
#!/bin/bash
#BSUB -J deepsnooze_${GROUP}_standard
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o ${GROUP}_standard.out
#BSUB -e ${GROUP}_standard.err

source ~/.bashrc

cd /zhome/96/e/167682/rml/DeepSnooze

uv run python -m deepsnooze.train model=cnn data=cv training=cv experiment_name=${GROUP} "wandb.notes='${NOTES}'"
EOF

# GROUP="cnn_deep_cv"

# bsub << EOF
# #!/bin/bash
# #BSUB -J deepsnooze_${GROUP}_deep
# #BSUB -q gpua10
# #BSUB -gpu "num=1:mode=exclusive_process"
# #BSUB -n 8
# #BSUB -R "rusage[mem=8GB]"
# #BSUB -W 24:00
# #BSUB -o ${GROUP}_deep.out
# #BSUB -e ${GROUP}_deep.err

# source ~/.bashrc

# cd /zhome/96/e/167682/rml/DeepSnooze

# uv run python -m deepsnooze.train model=cnn_deep data=cv training=cv experiment_name=${GROUP} "wandb.notes='${NOTES}'"
# EOF