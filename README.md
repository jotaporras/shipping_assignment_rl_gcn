# Shipping Assignment RL GCN

## Overview
This repository is a companion to the thesis titled "A Deep Reinforcement Learning Approach to Multistage Stochastic Network Flows for Distribution Problems. It contains the code for a RL environment that simulates Shipping Point Assignment (SPA), as well as the code to run the suite of experiments.

The thesis work, including experiment results can be found [here](https://github.com/jotaporras/shipping_assignment_rl_gcn/blob/main/Javier%20Porras%20Tesis%20para%20Biblioteca%20con%20Portada%20(1).pdf).


## Environment setup
To set up a development environment, the easiest way is to run the `gcp/setup.sh` script,  which will create the `ts_mcfrl` conda env and install the
local package containing the environment code.

To just reproduce the conda environment, run:

```shell
conda env create --name ts_mcfrl --file manual_env.yml
```

For the record, possibly all other environment YAML files were exported for reference of transitive dependencies, and probably don't work at all anyways.

## Reproducing the experiments
After setting up the W&B authentication and the conda env, the experiments are run from yaml files specifying the values for the sweep.
There are several sets of YAMLs, but the actual experiments from the thesis are from the YAMLs with the names: `python/src/experiments_seminar_2/final_experiments_v1/v2_thesis_run_{ENV_SIZE}__{AGENT_NAME}.yaml`.

They are called like so: 
```
wandb sweep $TESIS_DIR/python/src/experiments_seminar_2/final_experiments_v1/{SWEEP_YAML_FILE}
```
Then the `wandb agents` are triggered according to the output of that command.

Follow the logic in `python/src/experiments_seminar_2/final_experiments_v1/final_experiments_runner_v1.py` for details on how a training experiment is set up.


## Design
The environment was designed as an Open AI environment, although it wasn't verified that the interfaces were being fully respected. 
The training loop is implemented in PyTorch + PyTorch Lightning. The experiments are configured to log to a W&B workflow.

## Directories
| Directory   | Description                                                    |
|-------------|----------------------------------------------------------------|
| `python`    | Contains the source code for the simulator and experiments     |
| `notebooks` | Exploration notebooks                                          |
| `gcp`       | Shell scripts to set up environments to run experiments in GCP |


## Dependencies
- The Deep RL utilities in `python/src/dqn_common.py` were copied directly from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/lib/dqn_extra.py41, under an MIT License.
