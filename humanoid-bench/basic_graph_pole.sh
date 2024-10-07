#!/bin/bash
# graph 비교용 
# pole 테스크 walk, stair, hurdle 환경들 train script

# walk
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-pole_with_walk-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_pole_with_walk > logs/nohup_pole_with_walk.out &
# stair
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-pole_with_stair-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_pole_with_stair > logs/nohup_pole_with_stair.out &
# hurdle
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-pole_with_hurdle-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_pole_with_hurdle > logs/nohup_pole_with_hurdle.out &