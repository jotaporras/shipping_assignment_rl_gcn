#! /bin/bash


#sudo apt-get update
#sudo apt install git wget --yes
#sudo apt-get install bzip2 libxml2-dev --yes
#wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
#bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
#source ~/.bahsrc


git clone https://github.com/jotaporras/ts_mcfrl.git && cd ts_mcfrl && git fetch --all && git checkout other_gnn_architectures && git pull
conda env create --name ts_mcfrl --file manual_env.yml
conda activate ts_mcfrl

#TODO seems this has to be done manually, or maybe I can use api key multiple times?
#wandb login

export TESIS_DIR=/home/javier.porras/ts_mcfrl
# Installing the env package.
cd $TESIS_DIR/python/src/shipping_allocation
pip install -e .
cd $TESIS_DIR
export PYTHONPATH=/home/javier.porras/ts_mcfrl/python/src
conda update wandb
# Exporting conda env for faster install next time
#conda env export >> gcp_conda_env_n1.yml