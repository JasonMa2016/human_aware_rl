from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile


if __name__ == '__main__':
	file = load_pickle('data/ppo_poor_runs/ppo_bc_train_simple/config.pickle')
	print(file)