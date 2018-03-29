import logging
import time

import gym
import torch
import numpy as np
from setproctitle import setproctitle as ptitle
from torch.autograd import Variable

import environment
from utils import setup_logger, ensure_shared_grads
from network import MLP
from agent import Agent

RANDOM_SEED = 1

def test(args, nn):
    ptitle('Test Agent')
    
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    env = environment.make(args.env, args)

    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)

    player.model = MLP(player.env.observation_space.shape[0], player.env.action_space, args.n_frames)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    player.model.eval()
    max_score = 0

    while True:
        if player.done:
            player.model.load_state_dict(nn.state_dict())

        player.action_test()
        reward_sum += player.reward

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, reward {1}, average reward {2:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, reward_mean))

            if reward_sum >= max_score:
                max_score = reward_sum
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{}.dat'.format(args.model_save_dir))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            time.sleep(60)
            player.state = torch.from_numpy(state).float()


def train(rank, args, nn, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    
    env = environment.make(args.env, args)
    env.seed(RANDOM_SEED + rank)

    player = Agent(None, env, args, None)
    player.model = MLP(player.env.observation_space.shape[0], player.env.action_space, args.n_frames)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train()

    while True:
        player.model.load_state_dict(nn.state_dict())
        if player.done:
            player.cx = Variable(torch.zeros(1, 128))
            player.hx = Variable(torch.zeros(1, 128))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
            
        for step in range(args.n_steps):

            player.action_train()

            if player.done:
                break

        if player.done:
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()

        R = torch.zeros(1, 1)

        if not player.done:
            state = player.state
            value, _, _, _ = player.model(
                (Variable(state), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
  #          print(player.rewards[i])
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (0.01 * player.entropies[i].sum())

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, nn, gpu=False)
        optimizer.step()
        player.clear_actions()