# Lux-S3-public

Repository containing our code for training an agent for the NeurIPS 2024 - Lux AI Season 3 competition hosted by Kaggle https://www.kaggle.com/c/lux-ai-season-3. This project is a collaboration with Aurore Loisy and all authorship is shared.

A fully jittable jax agent can be assembled using several network architectures (resnet variants working the best) and several input and output formats defined via config yaml files. Separately from the neural network, the agent calculates and infers an ensemble of environment variables, game parameters, probabilities and other useful information called memory, all of which can be chosen as training input. The network is trained via PPO (from the Stoix library) against itself, a fixed opponent or a pool of opponents. The environment API follows the Jumanji convention.

The repo contains:
- `all_opponents`: storage directory for all jittable opponents
- `configs`: collection of yaml files that determine the hydra config
- `environment`: the Jumanji version of Lux where the opponent is part of the environment
- `external`: external libraries we use

  - `lux` is a copy of https://github.com/Lux-AI-Challenge/Lux-Design-S3/tree/main/src
  - `jumanji` contains selected modified files from https://github.com/instadeepai/jumanji/tree/main/jumanji needed to create a Jumanji environment
  - `stoix` contains selected modified files from https://github.com/EdanToledo/Stoix/tree/main/stoix, essentially the training algorithms
- `player`: the agent to train (self-contained neural network agent with hard-coded memory, used as part of the Kaggle submission)
- `scripts`: scripts for training the agent

## Installing Lux-S3-public

``` bash
conda create -n lux-s3 python==3.11
conda activate lux-s3
git clone git@github.com:vkrajnak/lux-s3.git
cd lux-s3
pip install .
pip install external/lux/
```

### Set input/output directory

The default path for inputs and outputs is the current working directory `$PWD`. 
It can be modified by setting the `LUX_IO_DIR` environment variable (e.g. add the following in your `~/.bashrc`):

``` bash
export LUX_IO_DIR="/path/to/dir"
```

Inside this directory, create an `outputs` directory where all outputs will be saved,
and a `configs` directory where you can have your own yaml config files.

### Setup tab completion with Hydra

If using bash:
``` bash
eval "$(evaluate -sc install=bash)"
```
otherwise check hydra documentation.

## Training an agent and using hydra

Run training with defaults
``` bash
python scripts/ff_ppo.py
```

Customize as needed for real experiments by overriding the default config (which is `configs/main/main_ff_ppo.yaml`)
``` bash
python scripts/ff_ppo.py env/reward=balanced opponents.use_selfplay=false
```

You can also use a custom config not already in the repo
``` bash
python scripts/ff_ppo.py env/reward=mynewreward
```
where `mynewreward.yaml` is your custom reward config located in your own `$LUX_IO_DIR/configs/env/reward` directory.

Logging a training with Neptune logging is disabled by default and can be enabled by
``` bash
python scripts/ff_ppo.py logger.use_neptune=true
```
In order for this to work, you need to set up your API key as an environment variable and specify your neptune project in `configs/logger/default.yaml`.

For training, you can preallocate 95% (instead of default 75%) of your GPU memory:
``` bash
conda env config vars set XLA_PYTHON_CLIENT_MEM_FRACTION=.95
```

## Useful

Run a "quick" test to see if the code runs 
``` bash
python scripts/ff_ppo.py arch=quicktest
```

Restart from checkpoint
``` bash
python scripts/ff_ppo.py loader=myloader [other_params]
```
where `$LUX_IO_DIR/configs/loader/myloader.yaml` is your loading config, which should have the same structure as `lux-s3-train/configs/loader/default.yaml`

By default the code uses all available cuda devices. You can specify which one(s) to use, e.g. device 1:
``` bash
CUDA_VISIBLE_DEVICES=1 python scripts/ff_ppo.py [...]
```

## Testing an agent

The trained agent is the directory `$LUX_IO_DIR/outputs/.../trained_agent...` where `...` is the ID of your training run. To test the agent, run 
``` bash
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```
using the official Lux repo https://github.com/Lux-AI-Challenge/Lux-Design-S3/tree/main
