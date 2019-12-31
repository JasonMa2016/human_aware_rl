import gym, time, os, seaborn, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from memory_profiler import profile

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorflow.saved_model import simple_save

PPO_DATA_DIR = '../../thesis_data/maml_ppo/'

ex = Experiment('PPO')
ex.observers.append(FileStorageObserver.create(PPO_DATA_DIR + 'ppo_exp'))

from overcooked_ai_py.utils import load_pickle, load_pkl, save_pickle, load_dict_from_file, profile
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_aware_rl.baselines_utils import get_vectorized_gym_env, create_model, update_model, save_baselines_model, load_baselines_model, get_agent_from_saved_model, overwrite_model, overwrite_variables
from human_aware_rl.utils import create_dir_if_not_exists, reset_tf, delete_dir_if_exists, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, DEFAULT_ENV_PARAMS, BC_SAVE_DIR
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH


# PARAMS
@ex.config
def my_config():

    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = True
    EX_NAME = "ppo_bc_train_simple_jason"

    if TIMESTAMP_DIR:
        SAVE_DIR = PPO_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PPO_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "ppo"

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = True

    # Choice among: bc_train, bc_test, sp, hm, rnd
    OTHER_AGENT_TYPE = "bc_train"

    # Human model params, only relevant if OTHER_AGENT_TYPE is "hm"
    HM_PARAMS = [True, 0.3]

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 30 if not LOCAL_TESTING else 2

    # Threshold for sparse reward before saving the best model
    SAVE_BEST_THRESH = 50

    # Every `VIZ_FREQUENCY` gradient steps, display the first 100 steps of a rollout of the agents
    VIZ_FREQUENCY = 50 if not LOCAL_TESTING else 10

    ##################
    # META PARAMS #
    ##################
    NUM_META_ITERATIONS = 20
    META_BATCH_SIZE = 10

    META_FACTOR = NUM_META_ITERATIONS * META_BATCH_SIZE

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 5e6 if not LOCAL_TESTING else 10000

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 800

    PPO_RUN_TOT_TIMESTEPS = PPO_RUN_TOT_TIMESTEPS // META_FACTOR
    # TOTAL_BATCH_SIZE = TOTAL_BATCH_SIZE // META_FACTOR

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 6 if not LOCAL_TESTING else 1

    # Calculating `batch size` as defined in baselines
    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads
    # This is nsteps what the heck is going on
    # Compare the two
    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 1e-3

    # Factor by which to reduce learning rate over training
    LR_ANNEALING = 1

    # Entropy bonus coefficient
    ENTROPY = 0.1

    # Value function coefficient
    VF_COEF = 0.1

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # None is default value that does no schedule whatsoever
    # [x, y] defines the beginning of non-self-play trajectories
    SELF_PLAY_HORIZON = None

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0

    # Whether mixing of self play policies
    # happens on a trajectory or on a single-timestep level
    # Recommended to keep to true
    TRAJECTORY_SELF_PLAY = True



    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3


    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = 'simple'
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    # Env params
    horizon = 400

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "RUN_TYPE": RUN_TYPE,
        "SEEDS": SEEDS,
        "LOCAL_TESTING": LOCAL_TESTING,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "mdp_generation_params": mdp_generation_params,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_META_ITERATIONS": NUM_META_ITERATIONS,
        "META_BATCH_SIZE": META_BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "LR_ANNEALING": LR_ANNEALING,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "SELF_PLAY_HORIZON": SELF_PLAY_HORIZON,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "OTHER_AGENT_TYPE": OTHER_AGENT_TYPE,
        "HM_PARAMS": HM_PARAMS,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "NETWORK_TYPE": NETWORK_TYPE,
        "SAVE_BEST_THRESH": SAVE_BEST_THRESH,
        "TRAJECTORY_SELF_PLAY": TRAJECTORY_SELF_PLAY,
        "VIZ_FREQUENCY": VIZ_FREQUENCY,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }

def save_ppo_model(model, save_folder):
    delete_dir_if_exists(save_folder, verbose=True)
    simple_save(
        tf.get_default_session(),
        save_folder,
        inputs={"obs": model.act_model.X},
        outputs={
            "action": model.act_model.action,
            "value": model.act_model.vf,
            "action_probs": model.act_model.action_probs
        }
    )


def configure_bc_agent(bc_save_dir, bc_model_path, gym_env, mlp, mdp):
    '''
    Configure the BC agent from its model path
    '''

    print("LOADING BC MODEL FROM: {}".format(bc_model_path))
    agent, bc_params = get_bc_agent_from_saved(bc_save_dir, bc_model_path)
    gym_env.use_action_method = True
    assert mlp.mdp == mdp
    agent.set_mdp(mdp)
    gym_env.other_agent = agent


def load_training_data(run_name, seeds=None):
    run_dir = PPO_DATA_DIR + run_name + "/"
    config = load_pickle(run_dir + "config")

    # To add backwards compatibility
    if seeds is None:
        if "NUM_SEEDS" in config.keys():
            seeds = list(range(min(config["NUM_SEEDS"], 5)))
        else:
            seeds = config["SEEDS"]

    train_infos = []
    for seed in seeds:
        train_info = load_pickle(run_dir + "seed{}/training_info".format(seed))
        train_infos.append(train_info)

    return train_infos, config

def get_ppo_agent(save_dir, seed=0, best=False):
    save_dir = PPO_DATA_DIR + save_dir + '/seed{}'.format(seed)
    config = load_pickle(save_dir + '/config')
    if best:
        agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])
    else:
        agent = get_agent_from_saved_model(save_dir + "/ppo_agent", config["sim_threads"])
    return agent, config

def match_ppo_with_other_agent(save_dir, other_agent, n=1, display=False):
    agent, agent_eval = get_ppo_agent(save_dir)
    ap0 = AgentPair(agent, other_agent)
    agent_eval.evaluate_agent_pair(ap0, display=display, num_games=n)

    # Sketch switch
    ap1 = AgentPair(other_agent, agent)
    agent_eval.evaluate_agent_pair(ap1, display=display, num_games=n)

def plot_ppo_run(name, sparse=False, limit=None, print_config=False, seeds=None, single=False):
    from collections import defaultdict

    # load all training data from an environment
    train_infos, config = load_training_data(name, seeds)

    if print_config:
        print(config)

    if limit is None:
        limit = config["PPO_RUN_TOT_TIMESTEPS"]

    num_datapoints = len(train_infos[0]['eprewmean'])

    prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
    ciel_data_idx = int(num_datapoints * prop_data)

    datas = []
    for seed_num, info in enumerate(train_infos):
        info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
        if single:
            plt.plot(info['xs'], info["ep_sparse_rew_mean"][:ciel_data_idx], alpha=1, label="Sparse{}".format(seed_num))
        datas.append(info["ep_sparse_rew_mean"][:ciel_data_idx])
    if not single:
        seaborn.tsplot(time=info['xs'], data=datas)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if single:
        plt.legend()


PYTHON_LAYOUT_NAME_TO_JS_NAME = {
    "unident_s": "asymmetric_advantages",
    "simple": "cramped_room",
    "random1": "coordination_ring",
    "random0": "random0",
    "random3": "random3"
}

JS_LAYOUT_NAME_TO_PYTHON_NAME = {v:k for k, v in PYTHON_LAYOUT_NAME_TO_JS_NAME.items()}


def load_workers(layout_name, type='train'):
    if type != 'train' and type!= 'test':
        raise ValueError("Invalid Human Data Type!")
    clean_trials = load_pkl('thesis_data/human/anonymized/clean_{}_trials.pkl'.format(type))
    current_clean_trials = clean_trials[clean_trials['layout_name'] == PYTHON_LAYOUT_NAME_TO_JS_NAME[layout_name]]
    workers = list(current_clean_trials['workerid_num'].unique())
    return workers

@ex.automain
# @profile
def ppo_run(params):

    create_dir_if_not_exists(params["SAVE_DIR"])
    save_pickle(params, params["SAVE_DIR"] + "config")

    #############
    # PPO SETUP #
    #############

    train_infos = []

    for seed in params["SEEDS"]:
        reset_tf()
        set_global_seed(seed)

        curr_seed_dir = params["SAVE_DIR"] + "seed" + str(seed) + "/"
        create_dir_if_not_exists(curr_seed_dir)

        save_pickle(params, curr_seed_dir + "config")

        print("Creating env with params", params)
        # Configure mdp

        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
        env = OvercookedEnv(mdp, **params["env_params"])
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True)

        # Configure gym env
        gym_env = get_vectorized_gym_env(
            env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
        )
        gym_env.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1 # 1
        gym_env.self_play_randomization = 1
        gym_env.trajectory_sp = params["TRAJECTORY_SELF_PLAY"] # True
        gym_env.update_reward_shaping_param(1 if params["mdp_params"]["rew_shaping_params"] != 0 else 0) # 1

        print("self_play_randomization: {}".format(gym_env.self_play_randomization))
        print("trajectory_sp: {}".format(gym_env.trajectory_sp))

        layout_name = params["mdp_params"]["layout_name"]

        bc_path = 'thesis_data/bc_runs/' + layout_name + '/' + 'bc_train' + '/'
        train_workers = load_workers(layout_name, "train")
        bc_agents = ['seed{}/worker{}'.format(i,j) for i in range(5) for j in train_workers]
        num_train_agents = len(bc_agents)

        # dummy agent to configure the environment
        bc_agent = bc_agents[0]
        configure_bc_agent(bc_path, bc_agent, gym_env, mlp, mdp)

        # Create meta model
        with tf.device('/device:GPU:{}'.format(params["GPU_ID"])):
            meta_model = create_model(gym_env, "ppo_agent", **params)

        params["CURR_SEED"] = seed

        # meta-parameters
        beta = params["LR"]
        num_meta_iterations = params["NUM_META_ITERATIONS"]
        meta_batch_size = params["META_BATCH_SIZE"]

        for i in range(num_meta_iterations):
            print(" ")
            print("ITERATION: {}".format(i))
            print(" ")

            # turn-off self-play after a while
            if i > num_meta_iterations * 0.1:
                gym_env.trajectory_sp = False

            meta_gradient = []
            meta_parameters = np.array(tf.trainable_variables(meta_model.scope+'/ppo2_model'))
            sampled_agent_indices = random.sample(range(0, num_train_agents), meta_batch_size)
            for index in sampled_agent_indices:
                current_agent = bc_agents[index]
                configure_bc_agent(bc_path, current_agent, gym_env, mlp, mdp)

                with tf.device('/device:GPU:{}'.format(params["GPU_ID"])):
                    agent_model = create_model(gym_env, current_agent+str(i), **params)

                overwrite_model(meta_model, agent_model)

                update_model(gym_env, agent_model, **params)
                agent_parameters = tf.trainable_variables(agent_model.scope+'/ppo2_model')
                current_gradient = np.array([agent_parameters[i] - meta_parameters[i] for i in range(len(agent_parameters))])
                # agent_model.sess.close()
                if meta_gradient == []:
                    meta_gradient = current_gradient
                else:
                    meta_gradient += current_gradient

            # This is vanilla SGD, could improve
            meta_parameters -= beta * meta_gradient

            # with tf.device('/device:GPU:{}'.format(params["GPU_ID"])):
            #     meta_model = create_model(gym_env, "ppo_agent", **params)
            overwrite_variables(meta_parameters, tf.trainable_variables(meta_model.scope+'/ppo2_model'))

            # print(meta_parameters)



        # Save model
        save_ppo_model(meta_model, curr_seed_dir + meta_model.agent_name)
        # print("Saved training info at", curr_seed_dir + "training_info")
        # save_pickle(train_info, curr_seed_dir + "training_info")
        # train_infos.append(train_info)

    return train_infos
