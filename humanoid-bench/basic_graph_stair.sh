#!/bin/bash
# graph 비교용 
# stair 테스크 walk, hurdle, pole 환경들 train script

# walk
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-stair_with_walk-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_stair_with_walk > logs/nohup_stair_with_walk.out &
# hurdle
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-stair_with_hurdle-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_stair_with_hurdle > logs/nohup_stair_with_hurdle.out &
# pole
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-stair_with_pole-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_stair_with_pole > logs/nohup_stair_with_pole.out &