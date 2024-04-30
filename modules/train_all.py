import sys
sys.path.append('.')
from modules.utils import get_config, set_seed
from modules.main import main

seed = 42



if __name__ == "__main__":

    model_names = ['model_initial', 'model_audio', 'model_comparation', 'model_combined', 'model_combined_sim', 'multimodal_model']

    for model_name in model_names:

        set_seed(seed)

        try:
            config_file = sys.argv[sys.argv.index('--config') + 1]
            config = get_config(config_file, model_name)
        except ValueError:
            print('Please specify a correct config file with --config')
            sys.exit(1)

        main(config)
