import multiprocessing
from argparse import ArgumentParser

import gym

from agent import Agent
from actor_critic import ActorCriticMaster, ActorCriticSlave

def arg_parser():
    """Argument Parser

    Returns:
        An arguments object with defined attributes,
        e.g., args.expert, args.env, ...
    """
    parser = ArgumentParser(prog="Robust Walking Controller")

    parser.add_argument("--mode",
        dest = "mode",
        help = "Selection running mode: [train, test]",
        type = str,
        required = True
    )
    parser.add_argument("--env",
        dest = "env",
        help = "Select OpenAI Environment",
        type = str,
        required = True
    )
    parser.add_argument("--model_name",
        dest = "model_name",
        help = "Learning model file name",
        type = str,
    )
    parser.add_argument("--save_path",
        dest = "save_path",
        help = "Learning model saving directory",
        type = str
    )
    parser.add_argument("--render",
        dest = "render",
        help = "Enable OpenAI Graphic Rendering or not",
        action='store_true'
    )

    args = parser.parse_args()

    if args.mode not in ["train", "test"]:
        print("Do NOT have such option %s" % args.mode)
        exit(-1)

    # if args.mode == "train" and not args.save_path:
    #     print("Must specify model saving directory")
    #     exit(-1)

    return args


def create_agents(env_id, sess):
    env = gym.make(env_id)
    master = ActorCriticMaster("master", env, sess)
    
    agents, slaves = [], []
    for i in range(multiprocessing.cpu_count()):
        agent_id = "agent_" + str(i)
        slave_id = "slave_" + str(i)

        slave = ActorCriticSlave(slave_id, env, sess, master)
        slaves.append(slave)
        agents.append( Agent(agent_id, env_id, slave) )

    env.close()
    return agents, master, slaves
