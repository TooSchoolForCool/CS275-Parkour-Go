# CS275 Final Project

## A Reinforcement Learning Approach for Locomotion

Group Members:

- Ziqi Yang
- Yunchu Zhang
- Wandi Cui
- Zeyu Zhang

*****



>  In this project, we are going to implement a physics-based character to play Parkour Game in which a character runs through a challenging sequence of terrain. The locomotion skills developed for our physics-based character will not only target flat terrain but also more complex terrains. In this project, our goal is to learn controllers that allow simulated characters to traverse through complex terrains with gaps, slopes, steps and walls using highly dynamic gaits. In order to learn such terrain-adaptive locomotion skills, a reinforcement learning approach and a evolution strategy will be adopted. 
>
> Results will be demonstrated in a physics-based character to navigate challenging sequences of terrain.



### Dependencies

Our program is developed and tested on Ubuntu 16.04 using Python 3. It may also work well on Ubuntu 14.04 and MacOS, but we do NOT guarantee it works on Windows...



Before running our code, several dependent packages have to be installed. Below we list the core packages we used in this project

- gym
- torch
- box2d-py
- glfw
- h5py
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



### File Structure

There are 2 folders in current directory:

- a3c: The  Asynchronous Advantage Actor-Critic algorithm is implemented to learn the motion controller
- evostra: An evolution strategy is implemented



### How to Run the Code

Please get into the directory a3v or evostra, a more detailed README file is included.