import numpy as np

from utils import set_style
from experiments.bc_experiments import *
from experiments.ppo_bc_experiments import *
from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile
from baselines_utils import *

reset_tf()

# best_bc_model_paths = load_pickle("data/bc_runs/best_bc_model_paths")

# print(best_bc_model_paths)

# ppo_bc_seeds = {
#     "bc_train": [9456, 1887, 5578, 5987,  516],
#     "bc_test": [2888, 7424, 7360, 4467,  184]
# }

# ppo_bc_model_paths = {
#     'bc_train': {
#         "simple": "ppo_bc_train_simple", 

# 	}
# }

def get_ppo_agent(ppo_dir, save_dir, seed, best=False):
    save_dir = ppo_dir + save_dir + '/seed{}'.format(seed)
    config = load_pickle(save_dir + '/config')
    if best:
        agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])
    else:
        agent = get_agent_from_saved_model(save_dir + "/ppo_agent", config["sim_threads"])
    return agent, config


# PPO AGENTS ARE TRAINED WITH THE BEST BC MODEL

# UNDERSTAND WHY IS THERE A BEST IN PPO_AGENT

#
seeds = [9456, 1887, 5578, 5987, 516]

# for each PPO agent trained with the BEST BC model
ppo_path_half = 'data/ppo_poor_runs/'
ppo_path_full = 'data/ppo_runs/'
ppo_paths = {"good":ppo_path_full, "bad": ppo_path_half}

performances = dict.fromkeys(['good','bad'], [])

print(BC_SAVE_DIR)

for seed in seeds:
	ppo_bc_train_path = 'ppo_bc_train_simple'

	for i in range(5):
		bc_model_path = 'simple_bc_test_seed' + str(i)
		agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_path)
		evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

# Figure out where these agents are trained from hehe
		for model in ppo_paths:
			agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_paths[model], ppo_bc_train_path, seed, best=False)

			ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=10, display=False)
			avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
			performances[model].append(avg_ppo_and_bc)
			print("BC model: {}, PPO seed: {}, BC seed: {}".format(model, seed, i))
			print(avg_ppo_and_bc)

np.save('test_bad_BC_model.npy', performances)
for model in performances:
	results = np.array(performances[model])
	print("BC model: {}, Average return: {}, std: {}".format(model, np.mean(results), np.std(results)))

