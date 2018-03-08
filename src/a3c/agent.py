import numpy as np
import gym

TOTAL_EPISODES = 0

class Agent(object):

    def __init__(self, agent_id, env_id, learner, visualize=False):
        self._agent_id = agent_id
        self._env_id = env_id
        self._learner = learner
        self._visualize = visualize


    def start_learning(self, max_episodes=10000, update_frequency=10):
        global TOTAL_EPISODES
        env = gym.make(self._env_id)

        while TOTAL_EPISODES < max_episodes:
            observation = env.reset()
            done = False
            steps = 0
            episode_reward = 0.0
            observations, actions, rewards = [], [], []

            while not done:
                action = self._learner.predict_action(observation)

                next_observation, reward, done, _ = env.step(action)

                if reward == -100:
                    reward = -2
                
                if self._visualize and steps % 30 == 0:
                    env.render()

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)

                if steps % update_frequency == 0 or done:
                    self._update_learner(done, next_observation, observations, actions, rewards)
                    observations, actions, rewards = [], [], []

                episode_reward += reward
                observation = next_observation
                steps += 1

            self._print_episode_info(env, TOTAL_EPISODES, episode_reward)
            TOTAL_EPISODES += 1

        env.close()
        print("[%s] Free Gym Environment Resources" % self._agent_id)


    def _print_episode_info(self, env, episodes, reward):
        # position = env.unwrapped.hull.position[0]
        is_good = "good" if reward >= 200 else "----"
        print("[%s] | %s | Episode: %-7d | Reward: %-5.1f" 
            % (self._agent_id, is_good, episodes, reward))


    def _update_learner(self, done, next_observation, observations, actions, rewards):
        value = 0 if done else self._learner.calc_value_func(next_observation)

        target_vals = []
        for r in rewards[::-1]:
            value = r + 0.999 * value
            target_vals.append(value)
        target_vals.reverse()

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        target_vals = np.vstack(target_vals)

        self._learner.update_master(observations, actions, target_vals)
        self._learner.pull_master()


def main():
    pass


if __name__ == '__main__':
    main()
