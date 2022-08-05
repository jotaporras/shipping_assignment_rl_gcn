#! /bin/bash

# Created November 17
# Usage "wandb agent command" num_agents
# It triggers N parallel runs of wandb agents for an already existing sweep.

# Constants
export PYTHONPATH=python/src
export CONDA_ENV_NAME=ts_mcfrl

# args
export WANDB_RUN_COMMAND=$1
export NUM_TASKS=$2

echo "Starting $NUM_TASKS runs of '$WANDB_RUN_COMMAND'"

# Running N parallel processes os wandb run
for ((i=1;i<=NUM_TASKS;i++)); do
  echo "Triggering wandb agent $i with command '$WANDB_RUN_COMMAND'"
  sleep 1s #to not collapse wandb server
  eval $WANDB_RUN_COMMAND &
done
echo "Now waiting for threads to converge"
wait
echo "Finished the wait"