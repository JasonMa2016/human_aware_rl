from experiments.ppo_bc_experiments import *
from imitation.behavioural_cloning import get_bc_agent_from_saved, DEFAULT_ENV_PARAMS, BC_SAVE_DIR, plot_bc_run
from overcooked_ai_py.utils import load_pickle, load_pkl, save_pickle, load_dict_from_file, profile

ppo_bc_seeds = {
    "bc_train": [9456, 1887, 5578, 5987,  516],
}

ppo_bc_model_paths = {
    'bc_train': {
        "simple": "ppo_bc_train_simple", 
        "unident_s": "ppo_bc_train_unident_s",
    }
}

bc_save_dir = 'data/bc_runs/'
layout_name = 'unident_s'
bc_model_path = '_bc_train_seed4/'
bc_model = load_pickle(bc_save_dir+layout_name+bc_model_path + 'bc_metadata.pickle')

# plot_runs_training_curves(ppo_bc_model_paths, ppo_bc_seeds, single=False, save=True)
plot_bc_run(bc_model['train_info'],100)