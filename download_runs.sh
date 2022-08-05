#! /bin/bash
# Download the runs associated with a sweep
# God bless this page https://linuxhint.com/bash_loop_list_strings/ for teaching me about string arrays

# To get the list of sweeps, use the API to get the set of sweeps from all runs of a project
# (replace the $USER/$PATH):
# import wandb;api=wandb.Api();print(set([r.sweep.id for r in api.runs("jotaporras/deleteme_sample_data_generator")]))
#TODO replace the hard coded sweeps with something based on https://stackoverflow.com/a/31405855
# OR maybe https://stackoverflow.com/a/5257398 is simpler.
# TODO still need to strip the {}, probably with sed.
#t="{'lbpmko6q', 'kemwdbf6', 'okg9aa6o', 'xp1qmpy1', 'jhv1jyf4'}"
#a=($(echo "$t" | tr ',' '\n'))
#echo "${a[1]}"

source activate ts_mcfrl_3
# Project where the sweeps live
PROJECT_NAME=official_experiments_v2
# Local path to download run data
#BASE_DATA_DIR=official_experiments_v2_take1
BASE_DATA_DIR=official_experiments_v2_take1_copy

#declare -a SWEEPS=('10gwlwlx' '1uoje1z6' '3vf0c8bq' '41apmz08' '6cs911ud' '6hxuzhps' '8ovl18vz' 'amekpypv' 'aqud3apa' 'fisyieyo' 'o8wn0wzx' 'oizzfdyv' 'r1jbqfe6' 'sgtdbp3t' 'sijsrnhe')
#declare -a SWEEPS=('edn80nqj')
#declare -a SWEEPS=('uf4mfwsb' 'twybi0uy')
#declare -a SWEEPS=('tsvsk6lf' 'amekpypv' '41apmz08' 'xy3w6kqd' 'fisyieyo' '10gwlwlx' 'ml8un3hf' 'aak7toq9')
#declare -a SWEEPS=('eerf46aa' 'fsz807jr' 'p1s5xi4s' 'ml8un3hf' 'tsvsk6lf')
declare -a SWEEPS=('8un3b8o9' '39qrneje' 'c1a2szrd')
#WARNING, old sweeps  will be deleted
#echo "Deleting old sweeps, you have 5 seconds to abort"
#sleep 5s
#rm -rf $BASE_DATA_DIR
for SWEEP in ${SWEEPS[@]};
do
  echo "Running script for $SWEEP"
  python python/src/experiment_utils/wandb_download_sweep_runs.py \
  --base_data_dir=$BASE_DATA_DIR \
  --sweep_id=$SWEEP \
  --project_name=$PROJECT_NAME \
  --entity=jotaporras
done


# Downloading all sweep data as of Nov 23.