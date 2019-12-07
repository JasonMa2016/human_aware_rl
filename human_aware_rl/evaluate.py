from utils import set_style
from experiments.bc_experiments import *
from experiments.ppo_bc_experiments import *
from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile

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


# PPO AGENTS ARE TRAINED WITH THE BEST BC MODEL

# UNDERSTAND WHY IS THERE A BEST IN PPO_AGENT

#
seeds = [9456, 1887, 5578, 5987, 516]

# for each PPO agent trained with the BEST BC model
for seed in seeds:
	ppo_bc_train_path = 'ppo_bc_train_simple'
	agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_bc_train_path, seed, best=False)

	for i in range(5):
		bc_model_path = 'simple_bc_test_seed' + str(i)
		print(bc_model_path)
		agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_path)
		evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

# Figure out where these agents are trained from hehe

		ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=10, display=False)
		avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
		print("PPO seed: {}, BC seed: {}".format(seed, i))
		print(avg_ppo_and_bc)