from experiments.ppo_bc_experiments import *
ppo_bc_seeds = {
    "bc_train": [9456, 1887, 5578, 5987,  516],
}

ppo_bc_model_paths = {
    'bc_train': {
        "simple": "ppo_bc_train_simple", 
        "unident_s": "ppo_bc_train_unident_s",
    }
}

plot_runs_training_curves(ppo_bc_model_paths, ppo_bc_seeds, single=False, save=True)