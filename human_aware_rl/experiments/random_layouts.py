import time
import numpy as np
from ray import tune
from sacred import Experiment

ex = Experiment('HyperparamSweep')

from overcooked_ai_py.utils import load_pickle, save_pickle

from human_aware_rl.utils import reset_tf
from human_aware_rl.ppo.ppo import ex as ex_ppo
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH

# PARAMS
@ex.config
def my_config():

    ##############
    # Run Params #
    ##############

    LOCAL_TESTING = False
    layout_name = None
    OTHER_AGENT_TYPE = 'sp'
    mdp_generation_params={'size_bounds': ([5, 5], [4, 4]),'prop_empty': [0.99, 1],'prop_feats': [0, 0.6]}
    PPO_RUN_TOT_TIMESTEPS=1e7 if not LOCAL_TESTING else 10000
    REW_SHAPING_HORIZON=0
    EX_NAME = None

    
    ###############
    # Tune Params #
    ###############

    LR = (1e-5, 1e-3)
    GAMMA = (0.95, 1)
    LAM = (0.95, 1)
    MAX_GRAD_NORM = (0.001, 1)
    CLIPPING = (0.001, 0.1)

    uniform_tune_params = [ 
        "GAMMA", 
        "LAM",
        "MAX_GRAD_NORM",
        "CLIPPING"
    ]
    
    log_uniform_tune_params = [
        "LR"
    ]
    
    params = dict(locals())

def train_ppo(config):
    from human_aware_rl.ppo.ppo import ex as ex_ppo, ppo_run, my_config

    tune_config_updates = {'TRACK_TUNE':True}
    tune_config_updates.update(config)

    run = ex_ppo.run(config_updates=tune_config_updates)
    train_info = run.result[0]

def loguniform(low=0, high=1, size=None):
    chosen_lr = np.exp(np.random.uniform(np.log(low), np.log(high), size))
    print(chosen_lr)
    return chosen_lr

@ex.automain
def hyperparam_run(params):

    search_space = {}
    for k, v in params.items():
        if k in params["uniform_tune_params"]:
            search_space[k] = tune.uniform(*v)
        elif k in params["log_uniform_tune_params"]:
            _v = v # For some reason this is necessary
            search_space[k] = tune.sample_from(lambda spec: loguniform(*_v))
        elif k not in ["uniform_tune_params", "log_uniform_tune_params"]:
            search_space[k] = v

    scheduler = tune.schedulers.AsyncHyperBandScheduler(
        metric="sparse_reward", 
        grace_period=50,
        mode="max",
        max_t=5000
    )

    analysis = tune.run(
        train_ppo,
        name="hyperparam_sweep" + time.strftime('%Y_%m_%d-%H_%M_%S'),
        config=search_space,
        scheduler=scheduler,
        num_samples=150,
        resources_per_trial={"cpu": 16, "gpu": 1},
        local_dir='data/tune/'
    )

    return analysis