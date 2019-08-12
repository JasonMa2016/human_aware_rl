from overcooked_ai_py.utils import load_pickle, save_pickle

from human_aware_rl.utils import reset_tf
from human_aware_rl.ppo.ppo import ex as ex_ppo
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH


from ray import tune
import time
import numpy as np
from human_aware_rl.ppo.ppo import ex as ex_ppo

# def fun(lr, grl):
#     def boi(t):
#         return lr * 10 + 0.1 * t + grl
#     return boi

# def train_mnist(config):
#     optimizer = fun(lr=config["lr"], grl=config["grl"])
#     for i in range(20):
#         time.sleep(np.random.uniform(0, 2))
#         acc = optimizer(i)
#         tune.track.log(mean_accuracy=acc)

def train_ppo(config):
    lr = config["lr"]
    run = ex_ppo.run(config_updates={'LOCAL_TESTING': True, 'layout_name': 'simple', 'OTHER_AGENT_TYPE': 'sp', 'LR': lr, 'PPO_RUN_TOT_TIMESTEPS':5e5 })
    # train_info = run.result[0]

search_space = {
    "lr": tune.uniform(0, 1)
}

# searcher = tune.suggest.BasicVariantGenerator()
# searcher.add_configurations()

analysis = tune.run(
    train_ppo, 
    name="example",
    config=search_space,
    scheduler=tune.schedulers.AsyncHyperBandScheduler(metric="mean_accuracy", grace_period=10, mode="max"),
    num_samples=2
)


# Just making sure seeding is working correctly and not changing actual outputs


# Uncomment to make current output standard output to check against
# save_pickle(train_info, 'data/testing/ppo_sp_train_info')