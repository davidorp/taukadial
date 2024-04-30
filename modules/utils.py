import yaml
import os
from dotmap import DotMap
import torch
import time
from tqdm import tqdm
import copy
import numpy as np
from torchmetrics.regression import R2Score, MeanSquaredError
import math
import wandb
from transformers import BertTokenizer


def get_descriptions():

    tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")

    english_picture_1 = 'Coming and Going'
    english_picture_2 = 'Cat on the Tree'
    english_picture_3 = 'Cookie Theft'
    chinese_picture_1 = 'nightmarket'
    chinese_picture_2 = 'park'
    chinese_picture_3 = 'daddy'

    english_pictures = [english_picture_1, english_picture_2, english_picture_3]
    chinese_pictures = [chinese_picture_1, chinese_picture_2, chinese_picture_3]

    english_captions = ["The top picture illustrates the beginning of the story, where there is a happy and enthusiastic family embarking on a trip, even the little boy and the dog are seen outside the window. The middle of the story captures the time spent at their destination. The final of the story is shown in the bottom picture, where the family is returning, appearing visibly tired and somewhat somber after a full day out.", 
                            "There is a cat stuck in the tree, and the girl is screaming for help. A man climbed the tree in an attempt to rescue the cat, but the ladder fell down. At the base of the tree, a dog is making noise. Someone has called the fire brigade, and they are on their way with another ladder to assist in the rescue.", 
                            "There is a mother who is drying dishes next to the sink in the kitchen. She is not paying attention and has left the tap on. As a result, water is overflowing from the sink. Meanwhile, two children are attempting to make cookies from a jar when their mother is not looking. One of the children, a boy, has climbed onto a stool to get up to the cupboard where the cookie jar is stored. The stool is rocking precariously. The other child, a girl, is standing next to the stool and has her hand outstretched ready to be given cookies."]
    
    chinese_captions = ["图片讲述的是台湾的传统夜市，有抓鱼游戏、卖香肠和赢香肠的传统骰子游戏。", 
                            "图片讲述的是台湾公园里的日常活动。有人在公园里泡茶、下棋，有人在公园里玩跷跷板、荡秋千、打羽毛球。", 
                            "这幅画描述的是一位父亲照顾他的孩子。父亲正在熨烫衣服，家里的狗发现小宝宝试图触摸插头。趁父亲不注意，狗迅速介入。与此同时，猫被吓了一跳，打翻了花瓶。"]
    
    english_input_ids = []
    enlgish_masks = []
    
    chinese_input_ids = []
    chinese_masks = []
    
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")

    for caption in english_captions:
        tokenized = tokenizer_en(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        input_id = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        english_input_ids.append(input_id)
        enlgish_masks.append(attention_mask)

    for caption in chinese_captions:
        tokenized = tokenizer_zh(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        input_id = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        chinese_input_ids.append(input_id)
        chinese_masks.append(attention_mask)

    return english_input_ids, enlgish_masks, chinese_input_ids, chinese_masks


def train(model, train_dataloader, valid_dataloader, lossfn, task, optimizer, lr_scheduler, num_epochs, model_name, early_stopping, early_stopping_patience, cross_val=False, num_cross_val = 0):

    wandb.init(project="TAUKADIAL")
    wandb.config = {"epochs": num_epochs}
    wandb.watch(model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    if cross_val:
        log = open('logs/' + model_name + '/train_stats_' + str(num_cross_val) + '.txt', "w")
    else:
        log = open('logs/' + model_name + '/train_stats.txt', "w")
    
    num_training_steps = num_epochs * len(train_dataloader)

    progress_bar = tqdm(range(num_training_steps))

    patience = 0
    best_value = 0
    rest_best_values = []
    best_epoch = 0
    best_weights = None
  
    for epoch in range(num_epochs):

        # Initialize variables to accumulate metrics
        total_true = [] if task == 'classification' else torch.tensor([]).to(device)
        total_pred = [] if task == 'classification' else torch.tensor([]).to(device)
        total_loss = 0

        progress_bar.set_description(f"Epoch {epoch + 1}") 
        
        log.write('Epoch {}:\n'.format(epoch + 1))
        model.train()
        init_time = time.time()
        for (waveform, sample_rate, audio_features, utterance, input_ids, attention_masks, picture_idx, language, mmses, labels) in train_dataloader:
            outputs = model(input_ids, attention_masks, picture_idx, language, audio_features)

            if task == 'classification':
                loss = lossfn(outputs, labels)
            elif task == 'regression':
                loss = lossfn(outputs, mmses)

            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # computtorche and store metrics
            total_loss += loss.item()

            if task == 'classification':
                _, predictions = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)
                predictions = predictions.cpu().numpy().astype(int)
                labels = labels.cpu().numpy().astype(int)

                # Accumulate true and predicted labels for overall metrics calculation
                total_true.extend(labels)
                total_pred.extend(predictions)
            elif task == 'regression':
                predictions = outputs.detach()
                labels = mmses.detach()

                # Accumulate true and predicted labels for overall metrics calculation
                total_true = torch.cat((total_true, labels), 0)
                total_pred = torch.cat((total_pred, predictions), 0)

            
            progress_bar.set_postfix(
                loss=float(loss),
            )

            progress_bar.update(1)

        # Compute overall metrics
        if task == 'classification':
            specificity, sensitivity, precision, F1, UAR = get_metrics_classification(total_true, total_pred)
        elif task == 'regression':
            r2_value, rmse = get_metrics_MSE(total_true, total_pred)


        # Print overall results
        log.write('train completed in: {} seconds\n'.format(time.time() - init_time))
        log.write("Overall Metrics:\n")
        log.write('loss: {}\n'.format(total_loss / len(train_dataloader)))
        
        if task == 'classification':
            log.write(f"Accuracy: {UAR}\n")
            log.write(f"F1 Score: {F1}\n")
            log.write(f"Specificity: {specificity}\n")
            log.write(f"Sensitivity: {sensitivity}\n")
            log.write(f"Precision: {precision}\n")
            wandb.log({"train_loss": total_loss / len(train_dataloader), "train_UAR": UAR, "train_F1": F1})
        elif task == 'regression':
            log.write(f"R2 Score: {r2_value}\n")
            log.write(f"RMSE: {rmse}\n")
            wandb.log({"train_loss": total_loss / len(train_dataloader), "train_RMSE": rmse, "train_R2": r2_value})


        log.write('----------------------------------------\n')
        log.write('Validation\n')
        validation_value, rest_values = evaluation(model, valid_dataloader, lossfn, task, log)

        if (task == 'classification' and validation_value > best_value) or (task == 'regression' and (validation_value < best_value or best_value == 0)):
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            log.write('Best model updated in epoch {}\n'.format(epoch + 1))
            best_value = validation_value
            rest_best_values = rest_values
            log.write('Best model updated value: {}\n'.format(best_value))
            patience = 0
        else:
            patience += 1
        

        if patience == early_stopping_patience and early_stopping:
            log.write('Early stopping in epoch {}\n'.format(epoch + 1))
            print('Early stopping in epoch {}'.format(epoch + 1))
            break
        
        log.write('----------------------------------------\n')


    # log.write line to separe output
    log.write('----------------------------------------\n')
    log.write('Training completed\n')
    if task == 'classification':
        log.write('Best validation accuracy: {}\n'.format(best_value))
        log.write('Best validation F1: {}\n'.format(rest_best_values[3]))
        log.write('Best validation specificity: {}\n'.format(rest_best_values[0]))
        log.write('Best validation sensitivity: {}\n'.format(rest_best_values[1]))
        log.write('Best validation precision: {}\n'.format(rest_best_values[2]))
    elif task == 'regression':
        log.write('Best validation RMSE: {}\n'.format(best_value))
        log.write('Best validation R2: {}\n'.format(rest_best_values))
    log.write('Best epoch: {}\n'.format(best_epoch))
    print('Loading best model weights')
    model.load_state_dict(best_weights)
    print('Best model loaded')

    log.close()

    return model, best_value, rest_best_values


def evaluation(model, dataloader, lossfn, task, log, test = False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Initialize variables to accumulate metrics
    total_true = [] if task == 'classification' else torch.tensor([]).to(device)
    total_pred = [] if task == 'classification' else torch.tensor([]).to(device)
    total_loss = 0
    
    model.eval()
    for (waveform, sample_rate, audio_features, utterance, input_ids, attention_masks, picture_idx, language, mmses, labels) in dataloader:   
        with torch.no_grad():
            outputs = model(input_ids, attention_masks, picture_idx, language, audio_features)

        if task == 'classification':
            loss = lossfn(outputs, labels)
        elif task == 'regression':
            loss = lossfn(outputs, mmses)

        # computtorche and store metrics
        total_loss += loss.item()

        if task == 'classification':
            _, predictions = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            predictions = predictions.cpu().numpy().astype(int)
            labels = labels.cpu().numpy().astype(int)

            # Accumulate true and predicted labels for overall metrics calculation
            total_true.extend(labels)
            total_pred.extend(predictions)
        elif task == 'regression':
            predictions = outputs.detach()
            labels = mmses.detach()

            # Accumulate true and predicted labels for overall metrics calculation
            total_true = torch.cat((total_true, labels), 0)
            total_pred = torch.cat((total_pred, predictions), 0)

    # Compute overall metrics
    if task == 'classification':
        specificity, sensitivity, precision, F1, UAR = get_metrics_classification(total_true, total_pred)
    elif task == 'regression':
        r2_value, rmse = get_metrics_MSE(total_true, total_pred)

    # Print overall results
    log.write("Overall Metrics:\n")
    log.write('loss: {}\n'.format(total_loss / len(dataloader)))
    
    if task == 'classification':
        log.write(f"Accuracy: {UAR}\n")
        log.write(f"F1 Score: {F1}\n")
        log.write(f"Specificity: {specificity}\n")
        log.write(f"Sensitivity: {sensitivity}\n")
        log.write(f"Precision: {precision}\n")
        if test:
            wandb.log({"test_loss": total_loss / len(dataloader), "test_UAR": UAR, "test_F1": F1})
        else:
            wandb.log({"validation_loss": total_loss / len(dataloader), "validation_UAR": UAR, "validation_F1": F1})
        return UAR, [specificity, sensitivity, precision, F1]
    elif task == 'regression':
        log.write(f"R2 Score: {r2_value}\n")
        log.write(f"RMSE: {rmse}\n")
        if test:
            wandb.log({"test_loss": total_loss / len(dataloader), "test_RMSE": rmse, "test_R2": r2_value})
        else:
            wandb.log({"validation_loss": total_loss / len(dataloader), "validation_RMSE": rmse, "validation_R2": r2_value})
        return rmse, r2_value

def test(model, dataloader, task, model_name):
    
    if task == 'classification':
        log = open('tests/' + model_name + '_task1.txt', "w")
        log.write('tkdname;dx\n')
    
    elif task == 'regression':
        log = open('tests/' + model_name + '_task2.txt', "w")
        log.write('tkdname;mmse\n')

    final_predictions = {}

    model.eval()
    for (audio_path, sample_rate, audio_features, utterance, input_ids, attention_masks, picture_idx, language, mmses, labels) in dataloader:   
        with torch.no_grad():
            outputs = model(input_ids, attention_masks, picture_idx, language, audio_features)

        audio_path = audio_path[0].split('/')[-1]

        if task == 'classification':
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy().astype(int)
            final_predictions[audio_path] = predictions[0]
        elif task == 'regression':
            predictions = outputs.detach().item()
            final_predictions[audio_path] = predictions


    # Read /dataset/test/taukadial_results_task1_attempt1.txt file
    with open('/dataset/test/taukadial_results_task1_attempt1.txt') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            name = line.split(';')[0]            
            if task == 'classification':
                log.write(name + ';' + ('NC' if final_predictions[name] == 0 else 'MCI') + '\n')
            else:
                log.write(name + ';' + str(final_predictions[name]) + '\n')


def final_evaluation(model, dataloader, lossfn, task, model_name):
    log = open('logs/' + model_name + '/test_stats.txt', "w")
    evaluation(model, dataloader, lossfn, task, log)
    log.close()


def get_config(config_file, model_name = 'nan'):

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    config = DotMap(config)
    if model_name != 'nan':
        config.model_name = model_name

    if config.task == 'regression':
        config.model_name = config.model_name + '_regression'

    # config.model_name += 'full'

    if not os.path.isdir('logs/'):
        os.mkdir('logs')

    if not os.path.isdir('logs/' + config.model_name):
        os.mkdir('logs/' + config.model_name)
    else:
        print('Model already exists')
        #exit()

    with open('logs/' + config.model_name + '/config.yaml', 'w') as f:
        yaml.dump(config, f)

    return config


def get_metrics_classification(y_true, y_pred):
    # Compute TN, TP, FN, FP
    TN = TP = FN = FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
    try:
        specificity = TN / (TN + FP)
    except ZeroDivisionError:
        specificity = 0
    try:
        sensitivity = TP / (TP + FN)
    except ZeroDivisionError:
        sensitivity = 0
    
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    
    try:
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    except ZeroDivisionError:
        F1 = 0
    
    try:
        UAR = (specificity + sensitivity) / 2
    except ZeroDivisionError:
        UAR = 0

    return specificity, sensitivity, precision, F1, UAR


def calculate_r_squared(y_obs, y_pred):
    mean_observed = torch.mean(y_obs)
    
    numerator = torch.sum((y_obs - y_pred)**2)
    denominator = torch.sum((y_obs - mean_observed)**2)
    
    r_squared = 1 - (numerator / denominator)
    
    return r_squared


def get_metrics_MSE(y_true, y_pred):

    y_true = y_true.cpu().squeeze()
    y_pred = y_pred.cpu().squeeze()

    r2_value = calculate_r_squared(y_true, y_pred)

    mean_squared_error_score = MeanSquaredError()
    mse = mean_squared_error_score(y_pred, y_true)
    rmse = math.sqrt(mse)

    return r2_value, rmse

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def get_model_statistics():
    directory = 'logs/'
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    for folder_name in folder_names:
        if '_regression' not in folder_name:
            result_file = open('logs/' + folder_name + '/cross_fold_summary.txt', "r")
            lines = result_file.readlines()
            UAR = np.array([])
            F1 = np.array([])
            specificity = np.array([])
            sensitivity = np.array([])
            precision = np.array([])
            for i in range(0, len(lines), 5):
                UAR = np.append(UAR, float(lines[i].split(' ')[-1]) * 100)
                F1 = np.append(F1, float(lines[i+1].split(' ')[-1]) * 100)
                specificity = np.append(specificity, float(lines[i+2].split(' ')[-1]) * 100)
                sensitivity = np.append(sensitivity, float(lines[i+3].split(' ')[-1]) * 100)
                precision = np.append(precision, float(lines[i+4].split(' ')[-1]) * 100)

            means = np.array([UAR.mean(), F1.mean(), specificity.mean(), sensitivity.mean(), precision.mean()])

            print(folder_name.replace('_', '\_'), ' & ', round(UAR.mean(), 2), ' $\pm$ ', round(UAR.std(), 1), ' & ', round(F1.mean(), 2), ' $\pm$ ', round(F1.std(), 1), ' & ', 
                    round(specificity.mean(), 2), ' $\pm$ ', round(specificity.std(), 1), ' & ', round(sensitivity.mean(), 2), ' $\pm$ ', round(sensitivity.std(), 1), ' & ', 
                    round(precision.mean(), 2), ' $\pm$ ', round(precision.std(), 1), ' & ', round(means.mean(), 2), ' $\pm$ ', round(means.std(), 1), '\\\\')


def get_model_statistics_regression():
    directory = 'logs/'
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    for folder_name in folder_names:
        if '_regression' in folder_name:
            result_file = open('logs/' + folder_name + '/cross_fold_summary.txt', "r")
            lines = result_file.readlines()
            R2 = np.array([])
            RMSE = np.array([])
            for i in range(0, len(lines), 2):
                R2 = np.append(R2, float(lines[i].split(' ')[-1]))
                RMSE = np.append(RMSE, float(lines[i+1].split(' ')[-1]))

            print(folder_name.replace('_', '\_'), ' & ', round(R2.mean(), 2), ' $\pm$ ', round(R2.std(), 1), ' & ', round(RMSE.mean(), 2), ' $\pm$ ', round(RMSE.std(), 1), '\\\\')
