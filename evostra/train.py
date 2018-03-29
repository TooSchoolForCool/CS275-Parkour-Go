#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import logging

from agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help="Training iteration times.")
    parser.add_argument('-s', '--save', type=str, default='weights.pkl',
                        help="Model weights file to save to.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    agent = Agent()

    logger.info('Training begins...')

    # Train the model.
    agent.train(iterations=args.iterations, filename=args.save)

    logger.info('Training finished...')


if __name__ == '__main__':
    main()
