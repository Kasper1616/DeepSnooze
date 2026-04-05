#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_cnn
### -- ask for number of cores (default: 1) --
#BSUB -n 7
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=2GB]"
### -- set the email address --
##BSUB -u s214786@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o train_gpu_%J.out
#BSUB -e train_gpu_%J.err
# -- end of LSF options --

nvidia-smi
module load cuda/11.6

# Standard
uv run -m deepsnooze.train training=default

# LoRA with rank 1, 3, 5
for k in 1 3 5; do
    uv run -m deepsnooze.train training=lora training.rank=$k
done

# Bayesian LoRA with rank 1, 3, 5
for k in 1 3 5; do
    uv run -m deepsnooze.train training=bayesian_lora training.rank=$k
done
