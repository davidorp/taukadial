import sys
sys.path.append('.')
from modules.dataset import get_dataloaders
from modules.utils import get_config, set_seed, get_descriptions, train
from modules.models import Model_comparation, Model_initial, Model_combined, Model_audio, Model_combined_sim, Model_audio_open_smile, Multimodal_model
from torch.optim import AdamW, Adam
from transformers import get_scheduler
import torch.nn as nn
import torch
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

seed = 42
set_seed(seed)
wandb.login()


def set_up(config, train_dataloader, device, fold = 0):

    english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks = get_descriptions()

    if config.model_name.replace('_regression', '') == 'model_initial':
        model = Model_initial().to(device)
    elif config.model_name.replace('_regression', '') == 'model_audio':
        model = Model_audio().to(device)
    elif config.model_name.replace('_regression', '') == 'model_audio_open_smile':
        model = Model_audio_open_smile().to(device)
    elif config.model_name.replace('_regression', '') == 'model_comparation':
        model = Model_comparation(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif config.model_name.replace('_regression', '') == 'model_combined':
        model = Model_combined(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif config.model_name.replace('_regression', '') == 'model_combined_sim':
        model = Model_combined_sim(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks).to(device)
    elif config.model_name.replace('_regression', '') == 'multimodal_model':
        model = Multimodal_model(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks, fold).to(device)
    else:
        print('Model not found')
        sys.exit(1)

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    #optimizer = Adam(model.parameters(), lr=config.train.learning_rate)
    
    if config.task == 'regression':
        lossfn = nn.MSELoss()
    elif config.task == 'classification':
        lossfn = nn.CrossEntropyLoss()

    num_training_steps = config.train.num_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # start a new experiment
    wandb.init(
        project="TAUKADIAL",
        name=(config.model_name) if not config.train.cross_validation else (config.model_name + '_' + str(fold)),
        config={
            "learning_rate": config.train.learning_rate,
            "architecture": config.model_name,
            "dataset": "TAUKADIAL",
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
            "task": config.task
        }
    )

    # capture a dictionary of hyperparameters with config
    wandb.watch(model)
    
    return model, optimizer, lossfn, lr_scheduler

def main(config):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config.train.cross_validation:
        dataset = get_dataloaders(device, config, True)
    else:
        train_dataloader, valid_dataloader = get_dataloaders(device, config)

    if config.model_name == 'model_audio':
        config.train.learning_rate = 0.1
        config.train.early_stopping_patience = 15
    
    if config.train.cross_validation:
        log = open('logs/' + config.model_name + '/cross_fold_summary.txt', "w")
        kf = KFold(n_splits=config.train.cross_validation_folds, shuffle=True, random_state=seed)
        for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
            # Split the data into training and testing sets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            # Create DataLoader for training and testing sets
            train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
            valid_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

            model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device, fold)

            model, best_value, rest_best_values = train(model, train_dataloader, valid_dataloader, lossfn, config.task, optimizer, lr_scheduler, config.train.num_epochs, 
                                                   config.model_name, config.train.early_stopping, config.train.early_stopping_patience, config.train.cross_validation, fold)
            print('fold:', fold, 'best_value:', best_value)
            log.write('fold: ' + str(fold) + ' best_value: ' + str(best_value) + '\n')
            if config.task == 'classification':
                log.write('Best F1: {}\n'.format(rest_best_values[3]))
                log.write('Best specificity: {}\n'.format(rest_best_values[0]))
                log.write('Best sensitivity: {}\n'.format(rest_best_values[1]))
                log.write('Best precision: {}\n'.format(rest_best_values[2]))
            else:
                log.write('Best R2: {}\n'.format(rest_best_values))
                
            torch.save(model.state_dict(), 'logs/' + config.model_name + '/model_fold_' + str(fold) + '.pth')
            print('Model saved')
            wandb.finish()
        log.close()
    else:
        model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device)
        model, best_value, rest_best_values = train(model, train_dataloader, valid_dataloader, lossfn, config.task, optimizer, lr_scheduler, config.train.num_epochs, config.model_name, 
                         config.train.early_stopping, config.train.early_stopping_patience)
        
        print('best_value:', best_value)
        if config.task == 'classification':
            print('Best F1: {}\n'.format(rest_best_values[3]))
            print('Best specificity: {}\n'.format(rest_best_values[0]))
            print('Best sensitivity: {}\n'.format(rest_best_values[1]))
            print('Best precision: {}\n'.format(rest_best_values[2]))
        else:
            print('Best R2: {}\n'.format(rest_best_values))
    
        # save model
        torch.save(model, 'logs/' + config.model_name + '/model.pt')
        print('Model saved')



if __name__ == "__main__":

    try:
        config_file = sys.argv[sys.argv.index('--config') + 1]
        config = get_config(config_file)
    except ValueError:
        print('Please specify a correct config file with --config')
        sys.exit(1)

    main(config)
