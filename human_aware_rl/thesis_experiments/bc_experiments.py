import copy
import numpy as np

from overcooked_ai_py.utils import load_pickle, load_pkl, save_pickle, mean_and_std_err
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair

from human_aware_rl.utils import reset_tf, set_global_seed, common_keys_equal
from human_aware_rl.imitation.behavioural_cloning import train_bc_agent, eval_with_benchmarking_from_saved, plot_bc_run, get_bc_agent_from_saved

DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

DEFAULT_DATA_PARAMS = {
    "train_mdps": ["simple"],
    "ordered_trajs": True,
    "human_ai_trajs": False,
    "data_path": "thesis_data/human/anonymized/clean_train_trials.pkl"
}

DEFAULT_BC_PARAMS = {
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {}, # Nothing to overwrite defaults
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {}
}

BC_SAVE_DIR = 'thesis_data/bc_runs/'

PYTHON_LAYOUT_NAME_TO_JS_NAME = {
    "unident_s": "asymmetric_advantages",
    "simple": "cramped_room",
    "random1": "coordination_ring",
    "random0": "random0",
    "random3": "random3"
}

JS_LAYOUT_NAME_TO_PYTHON_NAME = {v:k for k, v in PYTHON_LAYOUT_NAME_TO_JS_NAME.items()}

def train_bc_agent_from_hh_data(layout_name, agent_name, worker_id, num_epochs, lr, adam_eps, model):
    """
    Train a BC agent for each worker in a layout from human-human data
    """
    bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
    bc_params["data_params"]['train_mdps'] = [layout_name]
    bc_params["data_params"]['data_path'] = "thesis_data/human/player/{}/clean_{}_trials_worker_{}.pkl".format(layout_name, model, worker_id)
    bc_params["mdp_params"]['layout_name'] = layout_name
    bc_params["mdp_params"]['start_order_list'] = None

    model_save_dir = "thesis_data/bc_runs/"+ layout_name + "/" + agent_name
    return train_bc_agent(model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps), model_save_dir

def train_bc_models(all_params, seeds):
    """Train len(seeds) num of models for each layout"""

    models = ['train', 'test']
    for model_name in models:
        clean_trials = load_pkl('thesis_data/human/anonymized/clean_{}_trials.pkl'.format(model_name))

        for params in all_params:
            current_clean_trials = clean_trials[clean_trials['layout_name'] == PYTHON_LAYOUT_NAME_TO_JS_NAME[params['layout_name']]]
            workers = list(current_clean_trials['workerid_num'].unique())
            for worker_id in workers:
                for seed_idx, seed in enumerate(seeds):
                    set_global_seed(seed)
                    model, save_dir = train_bc_agent_from_hh_data(agent_name="bc_{}/seed{}/worker{}/".format(model_name, seed_idx, worker_id),
                     model=model_name, worker_id=worker_id, **params)
                    plot_bc_run(model.bc_info, params['num_epochs'], save_dir = save_dir + 'training_plot.png')

                    reset_tf()

def run_all_bc_experiments():
    # Train BC models
    seeds = [5415, 2652, 6440, 1965, 6647]
    num_seeds = len(seeds)

    params_simple = {"layout_name": "simple", "num_epochs": 40, "lr": 1e-3, "adam_eps":1e-8}
    params_unident = {"layout_name": "unident_s", "num_epochs": 40, "lr": 1e-3, "adam_eps":1e-8}
    params_random1 = {"layout_name": "random1", "num_epochs": 40, "lr": 1e-3, "adam_eps":1e-8}
    params_random0 = {"layout_name": "random0", "num_epochs": 40, "lr": 1e-3, "adam_eps":1e-8}
    params_random3 = {"layout_name": "random3", "num_epochs": 40, "lr": 1e-3, "adam_eps":1e-8}

    all_params = [params_simple, params_random1, params_unident, params_random0, params_random3]
    train_bc_models(all_params, seeds)


if __name__ == '__main__':
    run_all_bc_experiments()