import threading

import gym
import pickle
import numpy as np
import tensorflow as tf

from actor_critic import ActorCriticMaster, ActorCriticSlave
from utils import arg_parser
from utils import create_agents


def train(args):
    sess = tf.Session()
    coord = tf.train.Coordinator()
    agents, master, slaves = create_agents(args.env, sess)

    sess.run(tf.global_variables_initializer())

    agents_threads = []
    for agent in agents:
        cb = lambda : agent.start_learning(max_episodes=100, update_frequency=10)
        thread = threading.Thread(target=cb)
        thread.start()
        agents_threads.append(thread)

    coord.join(agents_threads)

    sess.close()


def test(learner, env_id, rollout):
    env = gym.make(env_id)

    observation = env.reset()
    done = False

    for i in range(rollout):
        total_reward = 0.0

        while not done:
            action = learner.predict_action(observation)
            observation, reward, done, _ = env.step(action)
            env.render()

            total_reward += reward

        print("[Episode %d] reward: %.1lf" % (i + 1, total_reward))

    env.close()


def main():
    args = arg_parser()

    if args.mode == "train":
        train(args)
    


if __name__ == '__main__':
    main()