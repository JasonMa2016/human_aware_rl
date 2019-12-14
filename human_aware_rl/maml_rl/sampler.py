import gym 
import torch 
import numpy as np
import multiprocessing as mp 

from maml_rl.envs.subproc_vec_env import SubprocVecEnv # UNDERSTAND HOW THIS WORKS
from maml_rl.episode import BatchEpisodes


def make_env(env_name,evaluation=False):
    """
    return the function that generates the environment.
    """
    def _make_env():
        original = ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0']
        if env_name in original:
            return gym.make(env_name)
        elif env_name == 'ML10':
            if evaluation:
                return ML10.get_test_tasks()
            return ML10.get_train_tasks()
        elif env_name == 'ML45':
            if evaluation:
                return ML45.get_test_tasks()
            return ML45.get_train_tasks(env_name)
        else:
            if evaluation:
                return ML1.get_test_tasks(env_name)
            return ML1.get_train_tasks(env_name)
    return _make_env

class BatchSampler(object):
    """
    """
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count()-1, evaluation=False):
        self.env_name = env_name
        self.batch_size = batch_size 
        self.num_workers = num_workers 

        self.queue = mp.Queue() 
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = make_env(env_name)()

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        """
        """
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset() 
        dones = [False]
        successes = []
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device).float()
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy() 
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions)
            # successes = []
            # for info in infos:
            #     if 'success' in info:
            #         successes.append(info['success'])
            #     else:
            #         successes.append()
            # print(len(observations), len(actions), len(rewards), len(successes))

            if all(dones):
                for info in infos:
                    if 'success' in info:
                        successes.append(info['success'])

            episodes.append(observations, actions, rewards, new_observations, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        # print(episodes.rewards.shape)
        return episodes, np.array(successes)

    def reset_task(self, task):
        """
        Reset the task to the task given. 
        """
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        """
        """
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks

