#!/bin/bash

#SBATCH -t 14:49:59
#SBATCH --account=lcls:tmol1043723
##SBATCH --reservation=lcls:earlyscience
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem 0
#SBATCH --partition=milano
# Configure psana2 parallelization
# Uncomment following line to enable running SMD0 on dedicated node
# source setup_hosts_openmpi.sh

source /sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_092225/setup_env.sh

python3 preproc.py $@