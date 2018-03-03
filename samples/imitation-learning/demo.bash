#!/bin/bash
set -eux
for e in Humanoid-v2 Walker2d-v2 Hopper-v2
do
    python run_expert.py experts/$e.pkl $e --render --num_rollouts=1
done
