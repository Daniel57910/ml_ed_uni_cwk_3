#!/bin/sh
#SBATCH -N 4	  # nodes requested
#SBATCH -n 4	  # tasks requested
#SBATCH --mem=12000  # memory in Mb
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=Teach-Standard
#SBATCH --mail-user=s0905577
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

source ~/.bashrc && conda activate mlp3 && torchrun ml_ed_uni_cwk_3/code/train_model.py -m att_v1

