# WITH TUNE
python experiments/random_layouts.py with layout_name="random" EX_NAME="random_layouts" OTHER_AGENT_TYPE="sp" mdp_generation_params="{'size_bounds': ([5, 5], [4, 4]),'prop_empty': [0.99, 1],'prop_feats': [0, 0.6]}" PPO_RUN_TOT_TIMESTEPS=1e7 REW_SHAPING_HORIZON=7.5e6

# NON-TUNE
python ppo/ppo.py with layout_name="random" EX_NAME="random_layouts" OTHER_AGENT_TYPE="sp" LR=1e-4 mdp_generation_params="{'size_bounds': ([5, 5], [4, 4]),'prop_empty': [0.99, 1],'prop_feats': [0, 0.6]}" PPO_RUN_TOT_TIMESTEPS=1e7 REW_SHAPING_HORIZON=7.5e6