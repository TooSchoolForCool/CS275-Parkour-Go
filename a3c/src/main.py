import os
# Extremely Important
# This line MUST in front the `import torch` if you want to use
# multithreading
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.multiprocessing as mp

import environment
from utils import arg_parser
from network import MLP, CONV
from shared_optimizer import SharedAdam
from trainer import test, train
from tester import evaluate


# Global Variables
RANDOM_SEED = 1


def main():
    args = arg_parser()

    if args.mode == "train":
        env = environment.make(args.env, args)
        if args.networks == "MLP":
            nn = MLP(env.observation_space.shape[0], env.action_space, args.n_frames)
        elif args.networks == "CONV":
            nn = CONV(args.n_frames, env.action_space)

        optimizer = SharedAdam(nn.parameters())

        threads = []
        thread = mp.Process(target=test, args=(args, nn))
        thread.start()
        threads.append(thread)

        for i in range(0, args.n_workers):
            thread = mp.Process(target=train, args=(i, args, nn, optimizer))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    elif args.mode == "test":
        evaluate(args)


if __name__ == '__main__':
    main()