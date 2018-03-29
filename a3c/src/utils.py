import argparse
import logging
import math
import json

import numpy as np
import torch
from torch.autograd import Variable


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def arg_parser():
    parser = argparse.ArgumentParser("Async Actor Critic Learning Algorithm")

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        required=True,
        help="Running Mode: [train, test]"
    )
    parser.add_argument(
        "--env",
        dest="env",
        type=str,
        required=True,
        help="Gym Environment ID, e.g., BipedalWalker-v2"
    )
    parser.add_argument(
        "--n_workers",
        dest="n_workers",
        type=int,
        default=4,
        help="Number of training workers, default = 4"
    )
    parser.add_argument(
        "--n_frames",
        dest="n_frames",
        type=int,
        default=1,
        help="Number of memory frames for LSTM"
    )
    parser.add_argument(
        "--log",
        dest="log",
        type=str,
        default="./log/",
        help="Output directory of logging file"
    )
    parser.add_argument(
        "--model_save_dir",
        dest="model_save_dir",
        type=str,
        default="./models/saved_model",
        help="Trained model save directory"
    )
    parser.add_argument(
        "--model_load_dir",
        dest="model_load_dir",
        type=str,
        default="./models/saved_model",
        help="Trained model load directory"
    )
    parser.add_argument(
        '--gamma',
        dest="gamma",
        type=float,
        default=0.99,
        help='Discount factor for rewards (default: 0.99)'
    )
    parser.add_argument(
        '--tau',
        dest="tau",
        type=float,
        default=1.00,
        help='parameter for GAE (default: 1.00)')
    parser.add_argument(
        "--n_steps",
        dest="n_steps",
        type=int,
        default=20,
        help='Number of forward steps in A3C (default: 20)'
    )
    parser.add_argument(
        "--rollout",
        dest="rollout",
        type=int,
        default=100,
        help="Number of rollouts in evaluation process"
    )
    parser.add_argument(
        "--render",
        dest="render",
        action='store_true',
        default=True,
        help="If enable rendering during the evaluation process"
    )
    parser.add_argument(
        "--networks",
        dest="networks",
        type=str,
        default="MLP",
        help="Type of networks (MLP, CONV)"
    )

    args = parser.parse_args()

    return args


def normal(x, mu, sigma):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        if not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.clone().cpu()