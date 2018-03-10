# Evostra-based model for BipedalWalker-v2

## Requirements:
- box2D
- evostra
- gym
- numpy
- pickle

## How to run
- Train:
    > $ python3 train.py -i [iterations] -s [save_path]
- Play:
    > $ python3 play.py -e [episodes] -l [load_model_file] -e
- Examples:
    > $ python 3 train.py -i 400 -s weights_i400.pkl

    > $ python 3 play.py -e 1 -l weights_0.pkl
    
    (weights_0.pkl contains a well-trained model with good performance for BipedalWalker-v2)

## How to modify the model
- All the evostra model related parameters are at the beginning of agent.py.
- If the AGENT_HISTORY_LENGTH is modified, the layer node numbers in model.py also need to be modified accordingly (24 * AGENT_HISTORY_LENGTH for BipedalWalker-v2).
- For other modifications, refer to the comments in codes.

## Evolution Strategy
- Evolutio Strategy (ES) is an optimization technique based on ideas of adaptation and evolution. You can learn more about it at https://blog.openai.com/evolution-strategies/
- A fast Evolution Strategy implementation in Python: **evostra** https://github.com/alirezamika/evostra
