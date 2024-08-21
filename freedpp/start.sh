#!/bin/bash 
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p /root/anaconda3 
echo "export PATH=$PATH:/root/anaconda3/bin" >> ~/.bashrc
source ~/.bashrc
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freedpp
