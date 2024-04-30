import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

final_dim = 2

class Model_initial(nn.Module):
    def __init__(self, middle_size = 256):
        super(Model_initial, self).__init__()
        self.chinese_model = BertModel.from_pretrained("bert-base-chinese")
        self.english_model = BertModel.from_pretrained("bert-base-uncased")

        # freeze the parameters
        for param in self.chinese_model.parameters():
            param.requires_grad = False
        for param in self.english_model.parameters():
            param.requires_grad = False
            
        self.chinese_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.chinese_model_encoder = nn.TransformerEncoder(self.chinese_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))

        self.english_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.english_model_encoder = nn.TransformerEncoder(self.english_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))
        
            
        self.chinese_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )

        self.english_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )
        
        self.init_weights()

    def init_weights(self):
        for layer in self.chinese_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.english_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_masks, picture_idx, languages, _):
        inputs_chinese = []
        attention_masks_chinese = []
        inputs_english = []
        attention_masks_english = []
        for i, language in enumerate(languages):
            if language == 'chinese':
                inputs_chinese.append(input_ids[i])
                attention_masks_chinese.append(attention_masks[i])
            else:
                inputs_english.append(input_ids[i])
                attention_masks_english.append(attention_masks[i])

        if len(inputs_chinese) > 0:
            inputs_chinese = torch.stack(inputs_chinese)
            attention_masks_chinese = torch.stack(attention_masks_chinese)
            chinese_output = self.chinese_model(input_ids=inputs_chinese, attention_mask=attention_masks_chinese)
            chinese_output = chinese_output.last_hidden_state
            chinese_output = self.chinese_model_encoder(chinese_output)
            chinese_output = torch.mean(chinese_output, dim=1)
            chinese_output = self.chinese_model_classifier(chinese_output)
        else:
            chinese_output = torch.tensor([]).to(input_ids.device)

        if len(inputs_english) > 0:
            inputs_english = torch.stack(inputs_english)
            attention_masks_english = torch.stack(attention_masks_english)
            english_output = self.english_model(input_ids=inputs_english, attention_mask=attention_masks_english)
            english_output = english_output.last_hidden_state
            english_output = self.english_model_encoder(english_output)
            english_output = torch.mean(english_output, dim=1)
            english_output = self.english_model_classifier(english_output)
        else:
            english_output = torch.tensor([]).to(input_ids.device)


        # Chinese outpus and english outputs, two different tensors, ids_chinese have the idxs of the chinese inputs
        final_outputs = []
        idx_english = 0
        idx_chinese = 0
        for language in languages:
            if language == 'chinese':
                final_outputs.append(chinese_output[idx_chinese])
                idx_chinese += 1
            else:
                final_outputs.append(english_output[idx_english])
                idx_english += 1

        output = torch.stack(final_outputs)
        return output
    

class Model_comparation(nn.Module):
    def __init__(self, english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks):
        super(Model_comparation, self).__init__()
        self.chinese_model = BertModel.from_pretrained("bert-base-chinese")
        self.english_model = BertModel.from_pretrained("bert-base-uncased")

        # freeze the parameters
        for param in self.chinese_model.parameters():
            param.requires_grad = False
        for param in self.english_model.parameters():
            param.requires_grad = False

        self.initialize_embeddings(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks)
        
        # Embeddings for the real english description and the receioved in the input
        # sizes = [batch_size, max_length, 768]
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, final_dim)
        )
        self.init_weights()


    def init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def initialize_embeddings(self, english_inputs, english_masks, chinese_inputs, chinese_masks):

        self.english_embeddings = []
        self.chinese_embeddings = []

        for i in range(len(english_inputs)):
            english_output = self.english_model(input_ids=english_inputs[i], attention_mask=english_masks[i])
            english_output = english_output.last_hidden_state
            self.english_embeddings.append(english_output[0])

        for i in range(len(chinese_inputs)):
            chinese_output = self.chinese_model(input_ids=chinese_inputs[i], attention_mask=chinese_masks[i])
            chinese_output = chinese_output.last_hidden_state
            self.chinese_embeddings.append(chinese_output[0])

    def forward(self, input_ids, attention_masks, picture_idx, languages, _):
        inputs_chinese = []
        attention_masks_chinese = []
        inputs_english = []
        attention_masks_english = []
        final_embeddings = []
        for i, language in enumerate(languages):
            if language == 'chinese':
                inputs_chinese.append(input_ids[i])
                attention_masks_chinese.append(attention_masks[i])
                final_embeddings.append(self.chinese_embeddings[picture_idx[i]-1])
            else:
                inputs_english.append(input_ids[i])
                attention_masks_english.append(attention_masks[i])
                final_embeddings.append(self.english_embeddings[picture_idx[i]-1])

        if len(inputs_chinese) > 0:
            inputs_chinese = torch.stack(inputs_chinese)
            attention_masks_chinese = torch.stack(attention_masks_chinese)
            chinese_output = self.chinese_model(input_ids=inputs_chinese, attention_mask=attention_masks_chinese)
            chinese_output = chinese_output.last_hidden_state
        else:
            chinese_output = torch.tensor([]).to(input_ids.device)

        if len(inputs_english) > 0:
            inputs_english = torch.stack(inputs_english)
            attention_masks_english = torch.stack(attention_masks_english)
            english_output = self.english_model(input_ids=inputs_english, attention_mask=attention_masks_english)
            english_output = english_output.last_hidden_state
        else:
            english_output = torch.tensor([]).to(input_ids.device)

        # Chinese outpus and english outputs, two different tensors, ids_chinese have the idxs of the chinese inputs
        final_outputs = []
        idx_english = 0
        idx_chinese = 0
        for language in languages:
            if language == 'chinese':
                final_outputs.append(chinese_output[idx_chinese])
                idx_chinese += 1
            else:
                final_outputs.append(english_output[idx_english])
                idx_english += 1

        output = torch.stack(final_outputs)
        final_embeddings = torch.stack(final_embeddings).to(input_ids.device)
        output = self.cosine_similarity(output, final_embeddings)
        output = self.classifier(output)
        return output


class Model_combined(nn.Module):
    def __init__(self, english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks, middle_size = 256):
        super(Model_combined, self).__init__()
        self.chinese_model = BertModel.from_pretrained("bert-base-chinese")
        self.english_model = BertModel.from_pretrained("bert-base-uncased")

        # freeze the parameters
        for param in self.chinese_model.parameters():
            param.requires_grad = False
        for param in self.english_model.parameters():
            param.requires_grad = False

        self.initialize_embeddings(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks)

        self.chinese_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.chinese_model_encoder = nn.TransformerEncoder(self.chinese_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))

        self.english_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.english_model_encoder = nn.TransformerEncoder(self.english_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))
            
        self.chinese_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )

        self.english_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )
        
        self.init_weights()

    def init_weights(self):
        for layer in self.chinese_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.english_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def initialize_embeddings(self, english_inputs, english_masks, chinese_inputs, chinese_masks):

        self.english_embeddings = []
        self.chinese_embeddings = []

        for i in range(len(english_inputs)):
            english_output = self.english_model(input_ids=english_inputs[i], attention_mask=english_masks[i])
            english_output = english_output.last_hidden_state
            self.english_embeddings.append(english_output[0])

        for i in range(len(chinese_inputs)):
            chinese_output = self.chinese_model(input_ids=chinese_inputs[i], attention_mask=chinese_masks[i])
            chinese_output = chinese_output.last_hidden_state
            self.chinese_embeddings.append(chinese_output[0])

    def forward(self, input_ids, attention_masks, picture_idx, languages, _):
        inputs_chinese = []
        attention_masks_chinese = []
        inputs_english = []
        attention_masks_english = []
        
        english_embeddings_batch = []
        chinese_embeddings_batch = []

        for i, language in enumerate(languages):
            if language == 'chinese':
                inputs_chinese.append(input_ids[i])
                attention_masks_chinese.append(attention_masks[i])
                chinese_embeddings_batch.append(self.chinese_embeddings[picture_idx[i]-1])
            else:
                inputs_english.append(input_ids[i])
                attention_masks_english.append(attention_masks[i])
                english_embeddings_batch.append(self.english_embeddings[picture_idx[i]-1])

        if len(inputs_chinese) > 0:
            inputs_chinese = torch.stack(inputs_chinese)
            attention_masks_chinese = torch.stack(attention_masks_chinese)
            chinese_output = self.chinese_model(input_ids=inputs_chinese, attention_mask=attention_masks_chinese)
            chinese_output = chinese_output.last_hidden_state
            chinese_output = self.chinese_model_encoder(chinese_output)
            chinese_embeddings_batch = torch.stack(chinese_embeddings_batch).to(input_ids.device)
            chinese_output = chinese_output + chinese_embeddings_batch
            chinese_output = torch.mean(chinese_output, dim=1)
            chinese_output = self.chinese_model_classifier(chinese_output)
        else:
            chinese_output = torch.tensor([]).to(input_ids.device)

        if len(inputs_english) > 0:
            inputs_english = torch.stack(inputs_english)
            attention_masks_english = torch.stack(attention_masks_english)
            english_output = self.english_model(input_ids=inputs_english, attention_mask=attention_masks_english)
            english_output = english_output.last_hidden_state
            english_output = self.english_model_encoder(english_output)
            english_embeddings_batch = torch.stack(english_embeddings_batch).to(input_ids.device)
            english_output = english_output + english_embeddings_batch
            english_output = torch.mean(english_output, dim=1)
            english_output = self.english_model_classifier(english_output)
        else:
            english_output = torch.tensor([]).to(input_ids.device)


        # Chinese outpus and english outputs, two different tensors, ids_chinese have the idxs of the chinese inputs
        final_outputs = []
        idx_english = 0
        idx_chinese = 0
        for language in languages:
            if language == 'chinese':
                final_outputs.append(chinese_output[idx_chinese])
                idx_chinese += 1
            else:
                final_outputs.append(english_output[idx_english])
                idx_english += 1

        output = torch.stack(final_outputs)
        return output
    
class Model_combined_sim(nn.Module):
    def __init__(self, english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks, middle_size = 256):
        super(Model_combined_sim, self).__init__()
        self.chinese_model = BertModel.from_pretrained("bert-base-chinese")
        self.english_model = BertModel.from_pretrained("bert-base-uncased")

        # freeze the parameters
        for param in self.chinese_model.parameters():
            param.requires_grad = False
        for param in self.english_model.parameters():
            param.requires_grad = False

        self.initialize_embeddings(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks)
        
        # Embeddings for the real english description and the receioved in the input
        # sizes = [batch_size, max_length, 768]
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.chinese_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.chinese_model_encoder = nn.TransformerEncoder(self.chinese_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))

        self.english_model_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.english_model_encoder = nn.TransformerEncoder(self.english_model_encoder_layer, num_layers=2, norm=nn.LayerNorm(768))
            
        self.chinese_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )

        self.english_model_classifier = nn.Sequential(
            nn.Linear(768, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )
        
        self.init_weights()

    def init_weights(self):
        for layer in self.chinese_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.english_model_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def initialize_embeddings(self, english_inputs, english_masks, chinese_inputs, chinese_masks):

        self.english_embeddings = []
        self.chinese_embeddings = []

        for i in range(len(english_inputs)):
            english_output = self.english_model(input_ids=english_inputs[i], attention_mask=english_masks[i])
            english_output = english_output.last_hidden_state
            self.english_embeddings.append(english_output[0])

        for i in range(len(chinese_inputs)):
            chinese_output = self.chinese_model(input_ids=chinese_inputs[i], attention_mask=chinese_masks[i])
            chinese_output = chinese_output.last_hidden_state
            self.chinese_embeddings.append(chinese_output[0])

    def forward(self, input_ids, attention_masks, picture_idx, languages, _):
        inputs_chinese = []
        attention_masks_chinese = []
        inputs_english = []
        attention_masks_english = []
        
        english_embeddings_batch = []
        chinese_embeddings_batch = []

        for i, language in enumerate(languages):
            if language == 'chinese':
                inputs_chinese.append(input_ids[i])
                attention_masks_chinese.append(attention_masks[i])
                chinese_embeddings_batch.append(self.chinese_embeddings[picture_idx[i]-1])
            else:
                inputs_english.append(input_ids[i])
                attention_masks_english.append(attention_masks[i])
                english_embeddings_batch.append(self.english_embeddings[picture_idx[i]-1])

        if len(inputs_chinese) > 0:
            inputs_chinese = torch.stack(inputs_chinese)
            attention_masks_chinese = torch.stack(attention_masks_chinese)
            chinese_output = self.chinese_model(input_ids=inputs_chinese, attention_mask=attention_masks_chinese)
            chinese_output = chinese_output.last_hidden_state
            chinese_output = self.chinese_model_encoder(chinese_output)
            chinese_embeddings_batch = torch.stack(chinese_embeddings_batch).to(input_ids.device)
            chinese_similarities = self.cosine_similarity(chinese_output, chinese_embeddings_batch) * 0.5
            chinese_output = torch.mean(chinese_output, dim=1)
            chinese_output = chinese_output + chinese_similarities
            chinese_output = self.chinese_model_classifier(chinese_output)
        else:
            chinese_output = torch.tensor([]).to(input_ids.device)

        if len(inputs_english) > 0:
            inputs_english = torch.stack(inputs_english)
            attention_masks_english = torch.stack(attention_masks_english)
            english_output = self.english_model(input_ids=inputs_english, attention_mask=attention_masks_english)
            english_output = english_output.last_hidden_state
            english_output = self.english_model_encoder(english_output)
            english_embeddings_batch = torch.stack(english_embeddings_batch).to(input_ids.device)
            english_similarities = self.cosine_similarity(english_output, english_embeddings_batch) * 0.5
            english_output = torch.mean(english_output, dim=1)
            english_output = english_output + english_similarities
            english_output = self.english_model_classifier(english_output)
        else:
            english_output = torch.tensor([]).to(input_ids.device)


        # Chinese outpus and english outputs, two different tensors, ids_chinese have the idxs of the chinese inputs
        final_outputs = []
        idx_english = 0
        idx_chinese = 0
        for language in languages:
            if language == 'chinese':
                final_outputs.append(chinese_output[idx_chinese])
                idx_chinese += 1
            else:
                final_outputs.append(english_output[idx_english])
                idx_english += 1

        output = torch.stack(final_outputs)
        return output
    
class Model_audio(nn.Module):
    def __init__(self):
        super(Model_audio, self).__init__()
        self.audio_model = nn.Sequential(
            nn.Linear(43, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, final_dim)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.audio_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, a,b,c,d, audio_features):
        output = self.audio_model(audio_features)
        return output
    
# Model for audio in opensmile, input 6373
class Model_audio_open_smile(nn.Module):
    def __init__(self):
        super(Model_audio_open_smile, self).__init__()
        self.audio_model = nn.Sequential(
            nn.Linear(6373, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, final_dim)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.audio_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, a,b,c,d, audio_features):
        output = self.audio_model(audio_features)
        return output

class Multimodal_model(nn.Module):
    def __init__(self, english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks, fold=0, middle_size = 128):
        super(Multimodal_model, self).__init__()

        self.text_model = Model_combined_sim(english_input_ids, english_attention_masks, chinese_input_ids, chinese_attention_masks)

        if final_dim == 1:
            self.text_model.load_state_dict(torch.load('logs/model_combined_sim_regression/model_fold_' + str(fold) + '.pth'))
        else:
            self.text_model.load_state_dict(torch.load('logs/model_combined_sim/model_fold_' + str(fold) + '.pth'))
        self.text_model.english_model_classifier = nn.Sequential(self.text_model.english_model_classifier[0])
        self.text_model.chinese_model_classifier = nn.Sequential(self.text_model.chinese_model_classifier[0])
        
        self.audio_model = Model_audio()
        if final_dim == 1:
            self.audio_model.load_state_dict(torch.load('logs/model_audio_regression/model_fold_' + str(fold) + '.pth'))
        else:
            self.audio_model.load_state_dict(torch.load('logs/model_audio/model_fold_' + str(fold) + '.pth'))
        
        self.audio_model.audio_model = nn.Sequential(self.audio_model.audio_model[0])


        self.mixed_classifier = nn.Sequential(
            nn.Linear(256 + 64, middle_size),
            nn.LayerNorm(middle_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(middle_size, final_dim)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.mixed_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_masks, picture_idx, languages, audio_features):
        text_output = self.text_model(input_ids, attention_masks, picture_idx, languages, None)
        audio_output = self.audio_model(None, None, None, None, audio_features)
        output = torch.cat((text_output, audio_output), dim=1)
        output = self.mixed_classifier(output)
        return output