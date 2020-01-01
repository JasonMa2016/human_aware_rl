import numpy as np

from experiments.bc_experiments import *
from experiments.ppo_bc_experiments import *
from overcooked_ai_py.utils import load_pickle, load_pkl, save_pickle, load_dict_from_file, profile
from baselines_utils import *

reset_tf()

# Currently Amazon Turk data has different names for some of the layouts
PYTHON_LAYOUT_NAME_TO_JS_NAME = {
    "unident_s": "asymmetric_advantages",
    "simple": "cramped_room",
    "random1": "coordination_ring",
    "random0": "random0",
    "random3": "random3"
}

JS_LAYOUT_NAME_TO_PYTHON_NAME = {v:k for k, v in PYTHON_LAYOUT_NAME_TO_JS_NAME.items()}

def get_ppo_agent(ppo_dir, save_dir, seed, best=False):
    save_dir = ppo_dir + save_dir + '/seed{}'.format(seed)
    config = load_pickle(save_dir + '/config')
    if best:
        agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])
    else:
        agent = get_agent_from_saved_model(save_dir + "/ppo_agent", config["sim_threads"])
    return agent, config

# seeds = [9456, 1887, 5578, 5987, 516]
seeds = [9456]

ppo_path_half = 'data/ppo_runs_half/'
ppo_paths = {"half": ppo_path_half}

# maml_ppo_path = 'thesis_data/maml_ppo/'
# ppo_paths = {"maml": maml_ppo_path}

dr_ppo_path = 'thesis_data/dr_ppo/'
ppo_paths = {"dr": dr_ppo_path}
layout_name = 'simple'
bc_mode = 'train'

performances = {}

for seed in seeds:
    ppo_bc_train_path = 'ppo_bc_train_' + layout_name
    # ppo_bc_train_path = 'maml_ppo_bc_train_' + layout_name
    for model in ppo_paths:
        agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_paths[model], ppo_bc_train_path, seed, best=False)
        if model not in performances:
            performances[model] = {}


        clean_trials = load_pkl('thesis_data/human/anonymized/clean_{}_trials.pkl'.format(bc_mode))
        current_clean_trials = clean_trials[clean_trials['layout_name'] == PYTHON_LAYOUT_NAME_TO_JS_NAME[layout_name]]
        workers = list(current_clean_trials['workerid_num'].unique())
        for worker_idx in workers:
            if worker_idx not in performances[model]:
                performances[model][worker_idx] = []
            for seed_idx in range(5):
                agent_name = 'bc_{}/seed{}/worker{}/'.format(bc_mode, seed_idx, worker_idx)
                bc_model_path = layout_name + "/" + agent_name 
                agent_bc_test, bc_params = get_bc_agent_from_saved('thesis_data/bc_runs/', bc_model_path)
                evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

                ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=10, display=False)
                avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
                # print("PPO model: {}, BC model: {}, PPO seed: {}, BC seed: {}".format(model, bc_model, seed, i))
                performances[model][worker_idx].append(avg_ppo_and_bc)


# np.save('thesis_data/test_MAML_PPO_BC_model_simple_train.npy', performances)

for model in performances:
    for worker_idx in performances[model]:
        results = np.array(performances[model][worker_idx])
        print("PPO model: {}, worker: {}, Average return: {}, std: {}".format(model, worker_idx, np.mean(results), np.std(results)))
