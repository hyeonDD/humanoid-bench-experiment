#!/bin/bash

LOG_DIR="/home/aiffelzero/zero/humanoid/humanoid-bench/tdmpc2/logs"

# hurdle
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_hurdle-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_hurdle > $LOG_DIR/nohup_walk.out &
# pole
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_pole-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_pole > $LOG_DIR/nohup_walk.out &
# stair
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_stair-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_stair > $LOG_DIR/nohup_walk.out &