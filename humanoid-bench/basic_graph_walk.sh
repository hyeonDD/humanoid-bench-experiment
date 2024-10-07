#!/bin/bash
# graph 비교용 
# walk 테스크 hurdle, stair,pole 환경들 train script

# hurdle
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_hurdle-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_hurdle > logs/nohup_walk_with_hurdle.out &
# pole
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_pole-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_pole > logs/nohup_walk_with_pole.out &
# stair
nohup python -m tdmpc2.train exp_name=tdmpc task=humanoid_h1-walk_with_stair-v0 seed=0 disable_wandb=False wandb_entity=zerobeak wandb_project=default_walk_with_stair > logs/nohup_walk_with_stair.out &