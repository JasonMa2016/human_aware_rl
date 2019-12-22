import gym 
import torch 
import numpy as np
import multiprocessing as mp 

from human_aware_rl.baselines_utils import RewardShapingEnv
from human_aware_rl.maml_rl.episode import BatchEpisodes
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved

from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.models import register

def make_env(base_env, env_name, featurize_fn=None):
    def _make_env():
        gym_env = gym.make(env_name)

        gym_env.custom_init(base_env, featurize_fn=featurize_fn, baselines=True)
        return gym_env
    return _make_env

class BatchSampler(object):
    """
    """
    def __init__(self, base_env, env_name, mdp, batch_size, num_workers=mp.cpu_count()-1):
        self.base_env = base_env
        self.env_name = env_name
        self.batch_size = batch_size 
        self.num_workers = num_workers 

        self.mdp = mdp 
        # self.mlp = mlp
        self.queue = mp.Queue() 
        self.envs = RewardShapingEnv(SubprocVecEnv([make_env(base_env, env_name, featurize_fn=lambda x: mdp.lossless_state_encoding(x))
         for _ in range(num_workers)], queue=self.queue))
        # self.envs = SubprocVecEnv([make_env(base_env, env_name, featurize_fn=lambda x: mdp.lossless_state_encoding(x))
        #  for _ in range(num_workers)], queue=self.queue)
        self.envs.self_play_randomization = 1
        self.envs.trajectory_sp = True
        self.envs.update_reward_shaping_param(1)

        self._env = make_env(base_env, env_name, featurize_fn=lambda x: mdp.lossless_state_encoding(x))()

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        """
        """
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)

        observations, batch_ids = self.envs.reset() 

        observations, agent_id = observations['both_agent_obs'], observations['other_agent_env_idx']
        assert(len(set(agent_id)) == 1)
        observations = observations[:,1-agent_id[0]]
        dones = [False]
        successes = []
        count = 0
        while (not all(dones)) or (not self.queue.empty()):
            print(count)
            count += 1
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device).float()
                observations_tensor = observations_tensor.view(observations_tensor.shape[0], int(np.prod(
                    observations_tensor.shape[1:])))
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
                # print(actions)
                # # get real action s
                # actions = np.stack([actions, np.zeros(len(actions),dtype=int)])
                # actions = actions.T
                other_actions = np.zeros(len(actions), dtype=int)
                print(actions)
                # print(list(actions))
                joint_action = [(actions[i], other_actions[i]) for i in range(len(actions))]
                print(joint_action)
            print("here?")
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(joint_action)
            print("done?")
            episodes.append(observations, actions, rewards, new_observations, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        # print(episodes.rewards.shape)
        return episodes, np.array(successes)

    def configure_bc_agent(self, bc_save_dir, bc_model_path):
        '''
        Configure the BC agent from its model path to the current overcooked environment
        '''
        # print("LOADING BC MODEL FROM: {}{}".format(bc_save_dir, bc_model_path))
        agent, bc_params = get_bc_agent_from_saved(bc_save_dir, bc_model_path)

        self.envs.use_action_method = True
        # assert self.mlp.mdp == self.mdp
        agent.set_mdp(self.mdp)
        self.envs.other_agent = agent

    # def reset_task(self, task):
    #     """
    #     Reset the task to the task given. 
    #     """
    #     tasks = [task for _ in range(self.num_workers)]
    #     reset = self.envs.reset_task(tasks)
    #     return all(reset)

    def sample_tasks(self, num_tasks):
        """
        """
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
