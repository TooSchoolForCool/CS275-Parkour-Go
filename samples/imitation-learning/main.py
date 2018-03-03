import gym
import pickle
import numpy as np

from argparse import ArgumentParser

from bc_model import BCModel


def arg_parser():
    """Argument Parser

    Returns:
        An arguments object with defined attributes,
        e.g., args.expert, args.env, ...
    """
    parser = ArgumentParser(prog="Behavioral Cloning Example")

    parser.add_argument("--expert",
        dest = "expert",
        help = "expert pickle file path",
        type = str,
        required = True
    )
    parser.add_argument("--env",
        dest = "env",
        help = "Select OpenAI Environment",
        type = str,
        required = True
    )
    parser.add_argument("--render",
        dest = "render",
        help = "Enable OpenAI Graphic Rendering or not",
        action='store_true'
    )
    parser.add_argument("--rollout",
        dest = "rollout",
        help = "Number of times of roll out",
        type = int,
        default = 1
    )

    return parser.parse_args()


def get_training_data(data_path):
    """Get traininng data set

    Args:
        data_path: [string] path to a pickle file in which expert data
            are stored. Expert data are a dictionary with two entries
            'observations' and 'actions'

    Returns:
        A tuple contains training data set
    """
    expert_data = pickle.load( open(data_path, "rb") )

    train_x = expert_data["observations"]
    train_y = expert_data["actions"]

    # flatten y 
    # [ [[y11, y12, y13, ...]], [[y21, y22, y23, ...]], ... ]
    # [ [y11, y12, y13, ...], [y21, y22, y23, ...], ... ]
    train_y = np.array([y[0] for y in train_y])

    return train_x, train_y


def start_env(args, bc_model):
    env = gym.make(args.env)

    rewards = []

    for i in range(args.rollout):
        observation = env.reset()
        done = False
        total_rewards = 0.0

        while not done:
            observation = np.array([observation])
            action = bc_model.predict(observation)
            observation, reward, done, _ = env.step(action)
            total_rewards += reward

            if args.render:
                env.render()

        rewards.append(total_rewards)

    print("totol rewards: %r" % rewards)
    print("mean of totol reward: %r" % np.mean(rewards))
    print("std of total reward %r" % np.std(rewards))


def main():
    args = arg_parser()

    train_x, train_y = get_training_data(args.expert)
    bc_model = BCModel(hidden_layers=[100, 100, 100, 40], learning_rate=5e-4)
    bc_model.train(train_x, train_y, n_epoch=50, batch_size=500)

    start_env(args, bc_model)


if __name__ == '__main__':
    main()
