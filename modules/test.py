import sys
sys.path.append('.')
from modules.dataset import get_dataloader_test
from modules.utils import test, set_seed, get_descriptions
from modules.models import Model_comparation, Model_initial, Model_combined, Model_audio, Model_combined_sim, Model_audio_open_smile, Multimodal_model
import torch

seed = 42
set_seed(seed)


def set_up(model_name, device, fold = 0):

    english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks = get_descriptions()

    if model_name.replace('_regression', '') == 'model_initial':
        model = Model_initial().to(device)
    elif model_name.replace('_regression', '') == 'model_audio':
        model = Model_audio().to(device)
    elif model_name.replace('_regression', '') == 'model_audio_open_smile':
        model = Model_audio_open_smile().to(device)
    elif model_name.replace('_regression', '') == 'model_comparation':
        model = Model_comparation(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif model_name.replace('_regression', '') == 'model_combined':
        model = Model_combined(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif model_name.replace('_regression', '') == 'model_combined_sim':
        model = Model_combined_sim(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif 'multimodal_model' in model_name:
        model = Multimodal_model(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks, fold).to(device)
    else:
        print('Model not found')
        sys.exit(1)

    
    return model


if __name__ == "__main__":

    model_to_test = 1

    if model_to_test == 1:
        fold = 1
        task = 'classification'
        model_name = 'multimodal_model'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_test(device, batch_size=1)
        model = set_up(model_name, device)
        model.load_state_dict(torch.load(f'logs/{model_name}/model_fold_{fold}.pth'))
    elif model_to_test == 2:
        fold = 0
        task = 'classification'
        model_name = 'multimodal_modelfull'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_test(device, batch_size=1)
        model = set_up(model_name, device)
        model.load_state_dict(torch.load(f'logs/{model_name}/model_fold_{fold}.pth'))
    elif model_to_test == 3:    
        fold = 2
        task = 'regression'
        model_name = 'multimodal_model_regression'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_test(device, batch_size=1)
        model = set_up(model_name, device)
        model.load_state_dict(torch.load(f'logs/{model_name}/model_fold_{fold}.pth'))
    elif model_to_test == 4:
        fold = 1
        task = 'regression'
        model_name = 'multimodal_model_regressionfull'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_test(device, batch_size=1)
        model = set_up(model_name, device)
        model.load_state_dict(torch.load(f'logs/{model_name}/model_fold_{fold}.pth'))
    
    test(model, dataloader, task, model_name)
