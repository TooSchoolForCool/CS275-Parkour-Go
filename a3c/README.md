# Asynchronous Advantage Actor-Critic (A3C)

In this project, we try to solve the [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) and [BipedalWalkerHardcore-v2](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/) problem by utilizing A3C algorithm. 



## Dependencies

Our program is developed and tested on Ubuntu 16.04 using Python 3. It may also work well on Ubuntu 14.04 and MacOS, but we do NOT guarantee it works on Windows...

Before running our code, several dependent packages have to be installed. Below we list the core packages we used in this project

- gym
- torch
- box2d-py
- glfw
- numpy
- Pillow
- PyOpenGL
- scipy
- six

To install OpenAI Gym, please check out the reference [HERE](https://github.com/openai/gym).

To install the PyTorch, please check out the reference [HERE](http://pytorch.org)

If you have further problems, please try following command

```bash
sudo pip3 install -r requirements.txt
```



## How to run our code

We have already provide some basic samples in the makefile.

Please try following commands to test our pre-trained BipedalWalker-v2 controller

```bash
make demo-simple
```

Please try following commands to test our pre-trained BipedalWalkerHardcore-v2 controller

```bash
make demo-hardcore
```



If you want to train your own model, here is an example

```bash
python3 ./src/main.py --mode=train --env="BipedalWalker-v2" --n_workers=4 --n_frames=1 --n_steps=20 --model_load_dir="./models/CS275-BipedalWalker-v2" --model_save_dir="./models/new-BipedalWalker-v2"
```

For more detail, please read the makefile or enter following command to learn what these optional arguments mean.

```
python3 main.py --help
```



