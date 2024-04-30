from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import csv
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BertTokenizer, AutoTokenizer
import torch.nn.functional as F
import math
import os

root_path = '/dataset/train/'

root_test_path = '/dataset/test/'

csv_file_path = 'groundtruth.csv'
csv_file_path_pos = 'pos_groundtruth.csv'
csv_file_path_pos_audio = 'pos_audio.csv'
csv_file_path_pos_audio_open_smile = 'pos_audio_opensmile.csv'
csv_file_path_pos_final = 'pos_groundtruth_final.csv'

def get_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=10,
        batch_size=16,
        return_timestamps=False,
        return_language=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe



class ChallengeDataset(Dataset):
    def __init__(self, audio_paths, audio_features, utterances, input_ids, attention_masks, languages, mses, labels, device):
        
        self.audio_paths = audio_paths
        self.audio_features = audio_features
        self.utterances = utterances
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.languages = languages
        self.mses = mses
        self.labels = labels
        self.device = device
                
    def __getitem__(self, ind):
        audio_path = self.audio_paths[ind]
        audio_features = self.audio_features[ind]
        utterance = self.utterances[ind]
        input_id = self.input_ids[ind]
        attention_mask = self.attention_masks[ind]
        language = self.languages[ind]
        mmse = self.mses[ind]
        label = self.labels[ind]

        if '-1.wav' in audio_path:
            picture = 1
        elif '-2.wav' in audio_path:
            picture = 2
        elif '-3.wav' in audio_path:
            picture = 3

        """
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        """
        waveform = 0
        sample_rate = 0
        waveform = self.audio_paths[ind]

        audio_features = torch.tensor(audio_features).to(self.device)
        input_id = input_id.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mmse = torch.tensor([int(mmse)]).float().to(self.device)
        label = label.to(self.device)

        return waveform, sample_rate, audio_features, utterance, input_id, attention_mask, picture, language, mmse, label

    def __len__(self):
        return len(self.audio_paths)
    


def get_preprocessed_transcript():

    pipe = get_models()

    with open(root_path + csv_file_path_pos, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=';')
        writer.writerow(['audio_path', 'transcript', 'language', 'mmse', 'dx'])

        # Read the csv file
        with open(root_path + csv_file_path, 'r') as read_file:
            reader = csv.reader(read_file)
            next(reader)
            for row in reader:
                tkdname,age,sex,mmse,dx = row
                audio_path = root_path + tkdname
                try:
                    result = pipe(audio_path)
                    if result['chunks'][0]['language'] == 'chinese':
                        result = pipe(audio_path, generate_kwargs={"language": "zh"})
                except Exception as e:
                    result = pipe(audio_path, generate_kwargs={"language": "zh"})

                writer.writerow([audio_path, result['text'], result['chunks'][0]['language'], mmse, dx])

def get_test_csv():

    with open(root_test_path + csv_file_path, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=',')

        writer.writerow(['audio_path', 'age', 'sex', 'mmse', 'dx'])
        directory = 'logs/'
        folder_names = [folder for folder in os.listdir(root_test_path) if 'wav' in folder]
        for audio_path in folder_names:
            writer.writerow([audio_path, 0,0,0,0])

def get_preprocessed_transcript_test():

    pipe = get_models()

    with open(root_test_path + csv_file_path_pos, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=';')
        writer.writerow(['audio_path', 'transcript', 'language', 'mmse', 'dx'])

        # Read the csv file
        with open(root_test_path + csv_file_path, 'r') as read_file:
            reader = csv.reader(read_file)
            next(reader)
            for row in reader:
                tkdname,age,sex,mmse,dx = row
                audio_path = root_test_path + tkdname
                try:
                    result = pipe(audio_path)
                    if result['chunks'][0]['language'] == 'chinese':
                        result = pipe(audio_path, generate_kwargs={"language": "zh"})
                except Exception as e:
                    result = pipe(audio_path, generate_kwargs={"language": "zh"})

                writer.writerow([audio_path, result['text'], result['chunks'][0]['language'], mmse, dx])

def add_age_sex_to_csv():
    with open(root_path + csv_file_path, 'r') as read_file:
        reader = csv.reader(read_file)
        next(reader)

        with open(root_path + csv_file_path_pos, 'r') as read_file_pos:
            reader_pos = csv.reader(read_file_pos, delimiter=';')
            next(reader_pos)

            with open(root_path + csv_file_path_pos_final, 'w', encoding='utf-8') as csv_file_writer:
                writer = csv.writer(csv_file_writer, delimiter=';')
                writer.writerow(['audio_path', 'transcript', 'language', 'age', 'sex', 'mmse', 'dx'])

                for row, row_pos in zip(reader, reader_pos):
                    audio_path, age, sex, mmse, dx = row
                    audio_path, transcript, language, mmse, dx = row_pos
                    writer.writerow([audio_path, transcript, language, age, sex, mmse, dx])
    

def get_audio_biomarkers():

    from opendbm import VerbalAcoustics, Speech

    indexes = ['audio_path', 'aco_int', 'aco_ff', 'aco_fm1', 'aco_fm2', 'aco_fm3', 'aco_fm4', 'aco_hnr', 'aco_gne', 'aco_jitter', 'aco_shimmer', 'aco_pauses', 'aco_voiceframe', 'aco_totvoiceframe', 
                'aco_voicepct', 'aco_mfcc1', 'aco_mfcc2', 'aco_mfcc3', 'aco_mfcc4', 'aco_mfcc5', 'aco_mfcc6', 'aco_mfcc7', 'aco_mfcc8', 'aco_mfcc9', 'aco_mfcc10', 'aco_mfcc11', 'aco_mfcc12', 
                'nlp_numSentences', 'nlp_singPronPerAns', 'nlp_singPronPerSen', 'nlp_pastTensePerAns', 'nlp_pastTensePerSen', 'nlp_pronounsPerAns', 'nlp_pronounsPerSen', 'nlp_verbsPerAns', 
                'nlp_verbsPerSen', 'nlp_adjectivesPerAns', 'nlp_adjectivesPerSen', 'nlp_nounsPerAns', 'nlp_nounsPerSen', 'nlp_sentiment_mean', 'nlp_mattr', 'nlp_wordsPerMin', 'nlp_totalTime']

    verbal_model = VerbalAcoustics()
    speech_model = Speech()

    functions = [verbal_model.get_audio_intensity, verbal_model.get_pitch_frequency, verbal_model.get_formant_frequency, 
             verbal_model.get_harmonic_noise, verbal_model.get_glottal_noise, verbal_model.get_jitter, verbal_model.get_shimmer, 
             verbal_model.get_pause_characteristics, verbal_model.get_voice_prevalence, verbal_model.get_mfcc,
             speech_model.get_speech_features]


    with open(root_path + csv_file_path_pos_audio, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=';')
        writer.writerow(indexes)

        with open(root_path + csv_file_path, 'r') as read_file:
            reader = csv.reader(read_file)
            next(reader)
            for row in reader:
                tkdname,age,sex,mmse,dx = row
                audio_path = root_path + tkdname

                verbal_model.fit(audio_path)
                speech_model.fit(audio_path)
                row = [audio_path]

                for function in functions:
                    try:
                        features = function().mean().to_frame()
                        for index, value in features.iterrows():
                            row.append(value[0])
                    except Exception as e:
                        print("Error in function: ", str(function))
                        print(e)
                        row.append(float('nan'))

                writer.writerow(row)

def get_audio_biomarkers_test():

    from opendbm import VerbalAcoustics, Speech

    indexes = ['audio_path', 'aco_int', 'aco_ff', 'aco_fm1', 'aco_fm2', 'aco_fm3', 'aco_fm4', 'aco_hnr', 'aco_gne', 'aco_jitter', 'aco_shimmer', 'aco_pauses', 'aco_voiceframe', 'aco_totvoiceframe', 
                'aco_voicepct', 'aco_mfcc1', 'aco_mfcc2', 'aco_mfcc3', 'aco_mfcc4', 'aco_mfcc5', 'aco_mfcc6', 'aco_mfcc7', 'aco_mfcc8', 'aco_mfcc9', 'aco_mfcc10', 'aco_mfcc11', 'aco_mfcc12', 
                'nlp_numSentences', 'nlp_singPronPerAns', 'nlp_singPronPerSen', 'nlp_pastTensePerAns', 'nlp_pastTensePerSen', 'nlp_pronounsPerAns', 'nlp_pronounsPerSen', 'nlp_verbsPerAns', 
                'nlp_verbsPerSen', 'nlp_adjectivesPerAns', 'nlp_adjectivesPerSen', 'nlp_nounsPerAns', 'nlp_nounsPerSen', 'nlp_sentiment_mean', 'nlp_mattr', 'nlp_wordsPerMin', 'nlp_totalTime']

    verbal_model = VerbalAcoustics()
    speech_model = Speech()

    functions = [verbal_model.get_audio_intensity, verbal_model.get_pitch_frequency, verbal_model.get_formant_frequency, 
             verbal_model.get_harmonic_noise, verbal_model.get_glottal_noise, verbal_model.get_jitter, verbal_model.get_shimmer, 
             verbal_model.get_pause_characteristics, verbal_model.get_voice_prevalence, verbal_model.get_mfcc,
             speech_model.get_speech_features]


    with open(root_test_path + csv_file_path_pos_audio, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=';')
        writer.writerow(indexes)

        with open(root_test_path + csv_file_path, 'r') as read_file:
            reader = csv.reader(read_file)
            next(reader)
            for row in reader:
                tkdname,age,sex,mmse,dx = row
                audio_path = root_test_path + tkdname

                verbal_model.fit(audio_path)
                speech_model.fit(audio_path)
                row = [audio_path]

                for function in functions:
                    try:
                        features = function().mean().to_frame()
                        for index, value in features.iterrows():
                            row.append(value[0])
                    except Exception as e:
                        print("Error in function: ", str(function))
                        print(e)
                        row.append(float('nan'))

                writer.writerow(row)


def get_audio_biomarkers_opensmile():

    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    with open(root_path + csv_file_path_pos_audio_open_smile, 'w', encoding='utf-8') as csv_file_writer:
        writer = csv.writer(csv_file_writer, delimiter=';')

        with open(root_path + csv_file_path, 'r') as read_file:
            reader = csv.reader(read_file)
            next(reader)
            for row in reader:
                tkdname,age,sex,mmse,dx = row
                audio_path = root_path + tkdname

                y = smile.process_file(audio_path)
                row = []

                for index in y:
                    row.append(y[index][0])
                writer.writerow(row)



def get_statistics():
    mmses = np.array([])
    mmses_mci = np.array([])
    mmses_nc = np.array([])
    english = 0
    chinese = 0
    mci = 0
    nc = 0

    mci_eng = 0
    mci_chi = 0

    nc_eng = 0
    nc_chi = 0

    ages = np.array([])
    males = 0
    females = 0

    mci_males = 0
    mci_females = 0

    age_mci_female = np.array([])
    age_mci_males = np.array([])

    with open(root_path + csv_file_path_pos_final, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        for row in reader:
            audio_path,text,language,age,sex,mmse,dx = row

            if sex == 'M':
                males += 1
                if dx == 'MCI':
                    mci_males += 1
                    age_mci_males = np.append(age_mci_males, int(age))
            else:
                females += 1
                if dx == 'MCI':
                    mci_females += 1
                    age_mci_female = np.append(age_mci_female, int(age))

            ages = np.append(ages, int(age))

            if language == 'english':
                english += 1
                if dx == 'MCI':
                    mci_eng += 1
                elif dx == 'NC':
                    nc_eng += 1
            elif language == 'chinese':
                chinese += 1
                if dx == 'MCI':
                    mci_chi += 1
                elif dx == 'NC':
                    nc_chi += 1

            if dx == 'MCI':
                mci += 1
                mmses_mci = np.append(mmses_mci, int(mmse))
            elif dx == 'NC':
                nc += 1
                mmses_nc = np.append(mmses_nc, int(mmse))

            mmses = np.append(mmses, int(mmse))
            
    print(mmses)
    print(f"English: {english}, Chinese: {chinese}")
    print(f"MCI: {mci}, NC: {nc}")
    print(f"Mean MMSE: {np.mean(mmses)}")
    print(f"Std MMSE: {np.std(mmses)}")

    print(f"Mean MMSE MCI: {np.mean(mmses_mci)}")
    print(f"Std MMSE MCI: {np.std(mmses_mci)}")

    print(f"Mean MMSE NC: {np.mean(mmses_nc)}")
    print(f"Std MMSE NC: {np.std(mmses_nc)}")

    print(f"English MCI: {mci_eng}, Chinese MCI: {mci_chi}")
    print(f"English NC: {nc_eng}, Chinese NC: {nc_chi}")

    print(f"Mean Age: {np.mean(ages)}")
    print(f"Std Age: {np.std(ages)})")
    print(f"Max Age: {np.max(ages)}")
    print(f"Min Age: {np.min(ages)}")

    print(f"Number of males: {males}")
    print(f"Number of females: {females}")

    print(f"Number of Males with MCI: {mci_males}")
    print(f"Number of Females with MCI: {mci_females}")

    print(f"Mean Age Males with MCI: {np.mean(age_mci_males)}")
    print(f"Mean Age Females with MCI {np.mean(age_mci_female)}")

    

def get_dataloaders(device, config, cross_validation=False):

    tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")

    with open(root_path + csv_file_path_pos, 'r') as file:
        with open(root_path + csv_file_path_pos_audio, 'r') as audio_file:

            reader = csv.reader(file, delimiter=';')
            audio_reader = csv.reader(audio_file, delimiter=';')
            
            next(reader)
            next(audio_reader)

            audio_paths = []
            utterances = []
            input_ids = []
            attention_masks = []
            languages = []
            mses = []
            labels = []
            audio_features = []

            for row, row_audio in zip(reader, audio_reader):
                audio_path,text,language,mmse,dx = row
                row_audio = row_audio[1:]
                data = [float(0) if math.isnan(float(d)) else float(d) for d in row_audio]
                if len(data) == 47:
                    data = data[:-6] + data[-2:]
                
                
                if language == 'chinese':
                    tokenized = tokenizer_zh(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                    input_id = tokenized['input_ids'][0]
                    attention_mask = tokenized['attention_mask'][0]

                else:
                    tokenized = tokenizer_en(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                    input_id = tokenized['input_ids'][0]
                    attention_mask = tokenized['attention_mask'][0]

                audio_paths.append(audio_path)
                audio_features.append(data)
                utterances.append(text)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                languages.append(language)
                mses.append(mmse)
                labels.append(dx) 

    labels = F.one_hot(torch.tensor([0 if label == 'NC' else 1 for label in labels]), num_classes=2).float()

    dataset = ChallengeDataset(audio_paths, audio_features, utterances, input_ids, attention_masks, languages, mses, labels, device)
    
    if cross_validation:
        return dataset
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.train.batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=config.train.batch_size)

    return train_dataloader, valid_dataloader


def get_dataloader_test(device, batch_size):

    tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")

    with open(root_test_path + csv_file_path_pos, 'r') as file:
        with open(root_test_path + csv_file_path_pos_audio, 'r') as audio_file:

            reader = csv.reader(file, delimiter=';')
            audio_reader = csv.reader(audio_file, delimiter=';')
            
            next(reader)
            next(audio_reader)

            audio_paths = []
            utterances = []
            input_ids = []
            attention_masks = []
            languages = []
            mses = []
            labels = []
            audio_features = []

            for row, row_audio in zip(reader, audio_reader):
                audio_path,text,language,mmse,dx = row
                row_audio = row_audio[1:]
                data = [float(0) if math.isnan(float(d)) else float(d) for d in row_audio]
                if len(data) == 47:
                    data = data[:-6] + data[-2:]
                
                
                if language == 'chinese':
                    tokenized = tokenizer_zh(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                    input_id = tokenized['input_ids'][0]
                    attention_mask = tokenized['attention_mask'][0]

                else:
                    tokenized = tokenizer_en(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                    input_id = tokenized['input_ids'][0]
                    attention_mask = tokenized['attention_mask'][0]

                audio_paths.append(audio_path)
                audio_features.append(data)
                utterances.append(text)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                languages.append(language)
                mses.append(mmse)
                labels.append(dx) 

    labels = F.one_hot(torch.tensor([0 if label == 'NC' else 1 for label in labels]), num_classes=2).float()

    dataset = ChallengeDataset(audio_paths, audio_features, utterances, input_ids, attention_masks, languages, mses, labels, device)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader
