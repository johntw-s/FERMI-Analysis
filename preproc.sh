#!/bin/bash
source /sdf/group/lcls/ds/ana/sw/conda2/rel/lcls2_092225/setup_env.sh

r_flag=''

while getopts 'r:' flag; do
  case "${flag}" in
    r) r_flag="${OPTARG}";;
  esac
done

echo $r_flag

# Change output for experiment
sbatch --output=${PWD}/logs/log_${r_flag}_%A.log preproc_backend.sh $@