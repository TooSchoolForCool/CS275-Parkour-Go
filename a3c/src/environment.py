from collections import deque

import gym
import numpy as np

def make(env_id, args=None):
    env = gym.make(env_id)
    env = EnvWrapper(env, args)

    return env


class EnvWrapper(gym.Wrapper):
    
    def __init__(self, env, args=None):
        gym.Wrapper.__init__(self, env)

        self._n_frames = args.n_frames
        self._frames = deque([], maxlen=self._n_frames)
        self._normalize = MaxMin()


    def __del__(self):
        self.env.close()


    def _reset(self):
        obs = np.float32( self.env.reset() )
        obs = self._normalize(obs)

        for _ in range(self._n_frames):
            self._frames.append(obs)

        return self._observation()


    def _observation(self):
        assert( len(self._frames) == self._n_frames )
        return np.stack(self._frames, axis=0)


    def _step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = np.float32(obs)
        obs = self._normalize(obs)

        self._frames.append(obs)

        return self._observation(), reward, done, info

    


class MaxMin(object):
    def __init__(self):
        self._max = 3.15
        self._min = -3.15
        self._new_max = 10
        self._new_min = -10


    def __call__(self, vec):
        """

        Args:
            vec: np.ndarray
        """
        new_interval = self._new_max - self._new_min
        old_interval = self._max - self._min

        obs = vec.clip(self._min, self._max)
        obs = (obs - self._min) / old_interval * new_interval + self._new_min

        return obs