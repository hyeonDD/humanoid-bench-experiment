#!/bin/bash
# graph 비교용 
# hurdle 테스크 walk, stair, pole 환경들 train script

# walk
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-hurdle_with_walk-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_hurdle_with_walk > logs/nohup_hurdle_with_walk.out &
# stair
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-hurdle_with_stair-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_hurdle_with_stair > logs/nohup_hurdle_with_stair.out &
# pole
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-hurdle_with_pole-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_hurdle_with_pole > logs/nohup_hurdle_with_pole.out &