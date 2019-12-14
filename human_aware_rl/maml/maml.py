from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_aware_rl.baselines_utils import get_vectorized_gym_env, create_model, update_model, save_baselines_model, load_baselines_model, get_agent_from_saved_model
from human_aware_rl.utils import create_dir_if_not_exists, reset_tf, delete_dir_if_exists, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, DEFAULT_ENV_PARAMS, BC_SAVE_DIR
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH

from human_aware_rl.meta_utils import *
from human_aware_rl.maml_rl.metalearner import MetaLearner 
from human_aware_rl.maml_rl.policies import CategoricalMLPPolicy
from human_aware_rl.maml_rl.baseline import LinearFeatureBaseline 

reset_tf()

if __name__ == "__main__":
    def run():
        layout_name= "unident_s"
        start_order_list = None

        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }


        horizon = 400

        mdp_params = {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        }

        env_params = {
            "horizon": horizon
        }

        NO_COUNTERS_PARAMS = {
                'start_orientations': False,
                'wait_allowed': False,
                'counter_goals': [],
                'counter_drop': [],
                'counter_pickup': [],
                'same_motion_goals': True
        }

        bc_path_full = 'data/bc_runs/'
        bc_path_half = 'data/bc_runs_poor/'
        bc_paths = {"good": bc_path_full, "bad": bc_path_half}

        # params = {"RUN_TYPE":'ppo',
        #           "sim_threads": 30}


        RUN_TYPE = "ppo"

        # Reduce parameters to be able to run locally to test for simple bugs
        LOCAL_TESTING = False

        # Choice among: bc_train, bc_test, sp, hm, rnd
        OTHER_AGENT_TYPE = "bc_train"

        # Human model params, only relevant if OTHER_AGENT_TYPE is "hm"
        HM_PARAMS = [True, 0.3]

        # GPU id to use
        GPU_ID = 1

        # List of seeds to run
        SEEDS = [0]

        # Number of parallel environments used for simulating rollouts
        sim_threads = 2

        # Threshold for sparse reward before saving the best model
        SAVE_BEST_THRESH = 50

        # Every `VIZ_FREQUENCY` gradient steps, display the first 100 steps of a rollout of the agents
        VIZ_FREQUENCY = 50 if not LOCAL_TESTING else 10

        ##############
        # PPO PARAMS #
        ##############

        # Total environment timesteps for the PPO run
        PPO_RUN_TOT_TIMESTEPS = 5e6 if not LOCAL_TESTING else 10000

        # How many environment timesteps will be simulated (across all environments)
        # for one set of gradient updates. Is divided equally across environments
        TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 800

        # Number of minibatches we divide up each batch into before
        # performing gradient steps
        MINIBATCHES = 6 if not LOCAL_TESTING else 1

        # Calculating `batch size` as defined in baselines
        BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

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
            # "EX_NAME": EX_NAME,
            # "SAVE_DIR": SAVE_DIR,
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

        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
        env = OvercookedEnv(mdp, **params["env_params"])
        # mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True) 

        # Configure gym env
        sampler = BatchSampler(env, 'Overcooked-v0', mdp, 3)
        # gym_env = get_vectorized_gym_env(
        #     env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
        # )

        sampler.configure_bc_agent(bc_paths['good'], 'unident_s_bc_train_seed' + str(1))


            # assume continuous actions
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(10, ) * 5)

        # Linear baseline 
        baseline = LinearFeatureBaseline(
            int(np.prod(sampler.envs.observation_space.shape)))

        metalearner = MetaLearner(sampler, policy, baseline)
        tasks = []
        for bc_model in bc_paths:
            for i in range(1):
                bc_model_path = 'unident_s_bc_train_seed' + str(i)
                tasks.append((bc_paths[bc_model], bc_model_path))

        episodes, grads, successes = metalearner.sample(tasks)
        # gym_env.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1
        # gym_env.trajectory_sp = params["TRAJECTORY_SELF_PLAY"]
        # gym_env.update_reward_shaping_param(1 if params["mdp_params"]["rew_shaping_params"] != 0 else 0)

        # for bc_model in bc_paths:
        #     for i in range(5):
        #         bc_model_path = 'unident_s_bc_train_seed' + str(i)
        #         print(bc_paths[bc_model], bc_model_path)
        #         configure_bc_agent(bc_paths[bc_model], bc_model_path, gym_env, mlp, mdp)
    run()