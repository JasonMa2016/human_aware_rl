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
seeds = [9456, 1887]

# for each PPO agent trained with the BEST BC model
ppo_path_half = 'data/ppo_runs_half/'
ppo_path_full = 'data/ppo_runs/'
ppo_path_quarter = 'data/ppo_runs_quarter/'

bc_path_full = 'data/bc_runs/'
bc_path_half = 'data/bc_runs_half/'
bc_path_quarter = 'data/bc_runs_quarter/'

ppo_paths = {"half": ppo_path_half, "quarter": ppo_path_quarter}
bc_paths = {"full": bc_path_full, "half": bc_path_half, "quarter": bc_path_quarter}

performances = {}

layout_name = 'unident_s'
for seed in seeds:
	ppo_bc_train_path = 'ppo_bc_train_' + layout_name
	for model in ppo_paths:
		agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_paths[model], ppo_bc_train_path, seed, best=False)
		if model not in performances:
			performances[model] = {}
		for bc_model in bc_paths:
			if bc_model not in performances[model]:
				performances[model][bc_model] = []
			for i in range(5):
				bc_model_path = layout_name + '_bc_test_seed' + str(i)
				agent_bc_test, bc_params = get_bc_agent_from_saved(bc_paths[bc_model], bc_model_path)
				evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

	# Figure out where these agents are trained from hehe
			# for model in ppo_paths:
			# 	if model not in performances:
			# 		performances[model] = []
			# 	agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_paths[model], ppo_bc_train_path, seed, best=False)

				ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=10, display=False)
				avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
				print("PPO model: {}, BC model: {}, PPO seed: {}, BC seed: {}".format(model, bc_model, seed, i))
				performances[model][bc_model].append(avg_ppo_and_bc)


np.save('test_PPO_BC_model_simple.npy', performances)

for model in performances:
	for bc_model in performances[model]:
		results = np.array(performances[model][bc_model])
		print("PPO model: {}, BC model: {}, Average return: {}, std: {}".format(model, bc_model, np.mean(results), np.std(results)))

