#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import logging

from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=1,
                        help="Playing episodes.")
    parser.add_argument('-l', '--load', type=str, default='weights.pkl',
                        help="Model weights file to load.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    agent = Agent()

    logger.info('Playing begins...')

    # Load and play.
    agent.load(args.load)
    agent.play(episodes=args.episodes)

    logger.info('Playing finished...')


if __name__ == '__main__':
    main()
