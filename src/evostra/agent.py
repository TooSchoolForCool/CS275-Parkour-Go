#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random

import gym
import numpy as np
import pickle
from evostra import EvolutionStrategy

from model import Model


class Agent:
    """The agent class."""

    ENV_ID = 'BipedalWalker-v2'
    # This is the number of the history obervations used in action prediction.
    AGENT_HISTORY_LENGTH = 1
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    # The following three parameters control the exlporation probabilities.
    # It starts with INITIAL_EXPLORATION, ends with FINAL_EXPLORATION after
    # EXLPORATION_DEC_STEPS steps.
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000


    def __init__(self):
        """Initialize the agent."""
        # Initialize the openai-gym environment.
        self.env = gym.make(self.ENV_ID)
        
        # uncomment following lines if you want to record the video
        # self.env = gym.wrappers.Monitor(self.env, "{}_monitor".format(self.ENV_ID),
        #     lambda episode_id: True, force=True)

        # Initialze the training model.
        self.model = Model()
        # Initialize the evolution strategy of evostra
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward,
                                    self.POPULATION_SIZE, self.SIGMA,
                                    self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION
        self.exploration_dec = self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS


    def train(self, iterations=100, print_step=1, filename='weights.pkl'):
        """Train the model."""
        self.es.run(iterations, print_step=print_step)
        self.save(filename)


    def load(self, filename='weights.pkl'):
        """Load the model weights from file."""
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp, encoding='bytes'))
        self.es.weights = self.model.get_weights()


    def save(self, filename='weights.pkl'):
        """Save the weights of current model into file."""
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)


    def play(self, episodes=1, render=True):
        """Play the agent for episodes."""
        self.model.set_weights(self.es.weights)

        for episode in range(episodes):
            total_reward = 0
            # Get the initial observation.
            observation = self.env.reset()
            # Fill the observation sequence with repeated initial obsercations
            # for AGENT_HISTORY_LENGTH times.
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    # Visualize.
                    self.env.render()
                action = self.get_predicted_action(sequence)
                # Get the results of the action.
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                # Shift the observation sequence to include the new one.
                sequence = sequence[1:]
                sequence.append(observation)

            print("total reward: ", total_reward)


    def get_predicted_action(self, sequence):
        """Get the model's predicted action based on sequence of states."""
        prediction = self.model.predict(np.array(sequence))
        return prediction


    def get_reward(self, weights):
        """Get the reward of the current model based on EPS_AVG times of
        tests."""
        total_reward = 0.0
        self.model.set_weights(weights)

        # Run tests for EPS_AVG times.
        for episode in range(self.EPS_AVG):
            # Get the initial observation.
            observation = self.env.reset()
            # Fill the observation sequence with repeated initial obsercations
            # for AGENT_HISTORY_LENGTH times.
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION,
                                       self.exploration - self.exploration_dec)
                # Randomize exploration.
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                # Get the results of the action.
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                # Shift the observation sequence to include the new one.
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward / self.EPS_AVG
