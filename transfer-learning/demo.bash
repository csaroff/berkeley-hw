#!/bin/bash
set -eux
for e in Hopper Ant HalfCheetah Humanoid Reacher Walker2d
# for e in Hopper
do
    python learner.py $e --num_rollouts 100 # --max_timesteps=333 render=True,
done