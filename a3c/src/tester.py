import logging

import gym
import torch
from torch.autograd import Variable

import environment
from utils import setup_logger
from network import MLP, CONV
from agent import Agent


def evaluate(args):
    torch.set_default_tensor_type('torch.FloatTensor')

    saved_state = torch.load(
        '{}.dat'.format(args.model_load_dir),
        map_location=lambda storage, loc: storage
    )

    log = {}
    setup_logger('{}_eval_log'.format(args.env), r'{0}{1}_eval_log'.format(
        args.log, args.env))
    log['{}_eval_log'.format(args.env)] = logging.getLogger(
        '{}_eval_log'.format(args.env))

    d_args = vars(args)
    for k in d_args.keys():
        log['{}_eval_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    env = environment.make("{}".format(args.env), args)
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)

    if args.networks == "MLP":
        player.model = MLP(env.observation_space.shape[0], env.action_space, args.n_frames)
    elif args.networks == "CONV":
        player.model = CONV(args.n_frames, env.action_space)

    if True:
        player.env = gym.wrappers.Monitor(
            player.env, "{}_monitor".format(args.env), lambda episode_id: True, force=True)

    player.model.load_state_dict(saved_state)

    player.model.eval()
    for i_episode in range(args.rollout):
        player.state = player.env.reset()
        player.state = torch.from_numpy(player.state).float()
        player.eps_len = 0
        reward_sum = 0
        while True:
            if args.render:
                if i_episode % 1 == 0:
                    player.env.render()

            player.action_test()
            reward_sum += player.reward

            if player.done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log['{}_eval_log'.format(args.env)].info(
                    "reward, {0}, average reward, {1:.4f}".format(reward_sum, reward_mean))
                break