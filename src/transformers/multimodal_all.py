# import libraries
import pickle
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel
import torch.nn as nn
import numpy as np 
from transformers import BertTokenizer
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW
import matplotlib.pyplot as plt
from udiva import UDIVA
from definitions import *
import logging

# setup GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# setup logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
# logger.setLevel("DEBUG")
logging.getLogger("transformers").setLevel(logging.ERROR)

# define models
class BertEncoder(nn.Module):
    def __init__(self, bert_model_name):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state
        return pooled_output, sequence_output

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(CrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input1, input2, attention_mask):
        query = input1
        key = input2
        value = input2
        attention_output, _ = self.self_attn(query, key, value, key_padding_mask=attention_mask)
        attention_output = attention_output.squeeze(0)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + input1)
        return attention_output

class CrossBERT(nn.Module):
    def __init__(self, bert_model_text, bert_model_landmarks, hidden_size, num_attention_heads, dropout_prob):
        super(CrossBERT, self).__init__()
        self.encoder1 = BertEncoder(bert_model_text)
        self.encoder2 = BertEncoder(bert_model_landmarks)
        self.linear_norm_1 = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.2), nn.Linear(768, 256))
        self.linear_norm_2 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(0.2), nn.Linear(256, 256))
        self.cross_attention = CrossAttention(hidden_size, num_attention_heads, dropout_prob)
        # freeze BERT weights
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
             param.requires_grad = False
        # self.regressor = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 5))    
        # self.regressor = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(0.2), nn.Linear(256, 5))
        self.regressor = nn.Sequential(nn.Linear(256, 5))

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        pooled_output1, sequence_output1 = self.encoder1(input_ids1, attention_mask1)
        pooled_output2, sequence_output2 = self.encoder2(input_ids2, attention_mask2)
        sequence_output1 = self.linear_norm_1(sequence_output1)
        sequence_output2 = self.linear_norm_2(sequence_output2)
        context = self.cross_attention(sequence_output1, sequence_output2, attention_mask2)
        # context = self.cross_attention(sequence_output2, sequence_output1, attention_mask1)
        output = context[:,0,:] # return only [CLS] token 
        output = self.regressor(output)
        return output

# util functions
def load_data_per_split(experiment_name, modality, vocab_size, split, type):
    cwd = os.getcwd()
    if vocab_size != 0:
        file_dir = os.path.join(cwd, "experiments", experiment_name, "data", f"{modality}_{split}_{type}_{vocab_size}.pkl")
        logger.debug(f"Reading from dir: {file_dir}")
    else: 
        file_dir = os.path.join(cwd, "experiments", experiment_name, "data", f"{modality}_{split}_{type}.pkl")
        logger.debug(f"Reading from dir: {file_dir}")
    with open(file_dir, "rb") as reader:
        data = pickle.load(reader)
    return data

def load_dataset(experiment_name, modality, split, type, vocab_size=0):
    if split == "all":
        data_train = load_data_per_split(experiment_name, modality, vocab_size, "train", type)
        data_val = load_data_per_split(experiment_name, modality, vocab_size, "val", type)
        data_test = load_data_per_split(experiment_name, modality, vocab_size, "test", type)
        logger.debug(f"Loaded train data with shape: {len(data_train)}")
        logger.debug(f"Loaded val data with shape: {len(data_val)}")
        logger.debug(f"Loaded test data with shape: {len(data_test)}")
    if type == "samples":
        return data_train, data_val, data_test
    elif type == "labels":
        return np.asarray(data_train).astype('float32'), np.asarray(data_val).astype('float32'), np.asarray(data_test).astype('float32')

def encode_corpus(data, tokenizer):
    encoded_corpus = tokenizer(text=data, add_special_tokens=True,
                                padding='max_length', truncation='longest_first',
                                max_length=300, return_attention_mask=True)
    return encoded_corpus

def return_ids_mask(encoded_corpus):
    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    return input_ids, attention_mask

# train, evaluate, and predict functions
def train(model, optimizer, loss_function, train_dataloader, device):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader): 
        batch_inputs_mod1, batch_masks_mod1, batch_labels_mod1, batch_inputs_mod2, batch_masks_mod2, batch_labels_mod2 = tuple(b.to(device) for b in batch)
        model.zero_grad()
        outputs = model(batch_inputs_mod1, batch_masks_mod1, batch_inputs_mod2, batch_masks_mod2)         
        loss = loss_function(outputs.squeeze(), batch_labels_mod1.squeeze())
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    return model, avg_train_loss

def evaluate(model, eval_dataloader, loss_function, device):
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch_inputs_mod1, batch_masks_mod1, batch_labels_mod1, batch_inputs_mod2, batch_masks_mod2, batch_labels_mod2 = tuple(b.to(device) for b in batch)
            outputs = model(batch_inputs_mod1, batch_masks_mod1, batch_inputs_mod2, batch_masks_mod2)      
            loss = loss_function(outputs.squeeze(), batch_labels_mod1.squeeze())
            total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(eval_dataloader)
    return model, avg_valid_loss

def predict(model, dataloader, device):
    model.eval() # sets the module in evaluation mode
    output = []
    for batch in dataloader:
        batch_inputs_mod1, batch_masks_mod1, batch_labels_mod1, batch_inputs_mod2, batch_masks_mod2, batch_labels_mod2 = tuple(b.to(device) for b in batch)
        model.zero_grad()
        with torch.no_grad():
            output += model(batch_inputs_mod1, batch_masks_mod1, batch_inputs_mod2, batch_masks_mod2).view(-1,5).tolist()
    return output

def aggregate_results(part_id, test_ids, results):
    test_ids = np.array(test_ids)
    indices = np.where(test_ids==part_id)
    indices = indices[0] # use the singleton
    part_results = np.take(results, indices, axis=0)
    mean = np.mean(part_results, axis=0)
    return mean

def create_dataloaders_multimodal(inputs_mod1, masks_mod1, labels_mod1, inputs_mod2, masks_mod2, labels_mod2, batch_size, shuffle=True):
    input_tensor_mod1 = torch.tensor(inputs_mod1)
    mask_tensor_mod1 = torch.tensor(masks_mod1, dtype=torch.float32)
    labels_tensor_mod1 = torch.tensor(labels_mod1)
    input_tensor_mod2 = torch.tensor(inputs_mod2)
    mask_tensor_mod2 = torch.tensor(masks_mod2, dtype=torch.float32)
    labels_tensor_mod2 = torch.tensor(labels_mod2)
    dataset = TensorDataset(input_tensor_mod1, mask_tensor_mod1, labels_tensor_mod1, input_tensor_mod2, mask_tensor_mod2, labels_tensor_mod2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":

    # load train data for all modalities
    text_train, text_val, text_test = load_dataset(experiment_name="multimodal", modality="text", split="all", type="samples")
    audio_train, audio_val, audio_test = load_dataset(experiment_name="multimodal", modality="audio", split="all", type="samples", vocab_size=128)
    landmarks_train, landmarks_val, landmarks_test = load_dataset(experiment_name="multimodal", modality="landmarks", split="all", type="samples", vocab_size=1024)

    # load train labels for all modalities
    text_train_labels, text_val_labels, text_test_labels = load_dataset(experiment_name="multimodal", modality="text", split="all", type="labels")
    audio_train_labels, audio_val_labels, audio_test_labels = load_dataset(experiment_name="multimodal", modality="audio", split="all", type="labels", vocab_size=128)
    landmarks_train_labels, landmarks_val_labels, landmarks_test_labels = load_dataset(experiment_name="multimodal", modality="landmarks", split="all", type="labels", vocab_size=1024)

    # set pretrained model directories
    cwd = os.getcwd()

    experiment_dir_01  = "XUL04"
    experiment_dir_02 = "XUL05"
    model_dir_audio = os.path.join(cwd, "experiments", experiment_dir_01)
    model_dir_landmarks = os.path.join(cwd, "experiments", "multimodal", experiment_dir_02)
    model_dir_text = "dccuchile/bert-base-spanish-wwm-cased"

    # load tokenizers
    audio_tokenizer = BertTokenizer.from_pretrained(model_dir_audio)
    landmarks_tokenizer = BertTokenizer.from_pretrained(model_dir_landmarks)
    text_tokenizer = BertTokenizer.from_pretrained(model_dir_text)

    # encode corpus
    # text
    encoded_corpus_text_train = encode_corpus(data=text_train, tokenizer=text_tokenizer)
    encoded_corpus_text_val = encode_corpus(data=text_val, tokenizer=text_tokenizer)
    encoded_corpus_text_test = encode_corpus(data=text_test, tokenizer=text_tokenizer)
    # audio
    encoded_corpus_audio_train = encode_corpus(data=audio_train, tokenizer=audio_tokenizer)
    encoded_corpus_audio_val = encode_corpus(data=audio_val, tokenizer=audio_tokenizer)
    encoded_corpus_audio_test = encode_corpus(data=audio_test, tokenizer=audio_tokenizer)
    # landmarks
    encoded_corpus_landmarks_train = encode_corpus(data=landmarks_train, tokenizer=landmarks_tokenizer)
    encoded_corpus_landmarks_val = encode_corpus(data=landmarks_val, tokenizer=landmarks_tokenizer)
    encoded_corpus_landmarks_test = encode_corpus(data=landmarks_test, tokenizer=landmarks_tokenizer)

    # compute attention masks and ids
    # text
    input_ids_text_train, attention_mask_text_train = return_ids_mask(encoded_corpus_text_train)
    input_ids_text_val, attention_mask_text_val = return_ids_mask(encoded_corpus_text_val)
    input_ids_text_test, attention_mask_text_test = return_ids_mask(encoded_corpus_text_test)
    # audio
    input_ids_audio_train, attention_mask_audio_train = return_ids_mask(encoded_corpus_audio_train)
    input_ids_audio_val, attention_mask_audio_val = return_ids_mask(encoded_corpus_audio_val)
    input_ids_audio_test, attention_mask_audio_test = return_ids_mask(encoded_corpus_audio_test)
    # landmarks
    input_ids_landmarks_train, attention_mask_landmarks_train = return_ids_mask(encoded_corpus_landmarks_train)
    input_ids_landmarks_val, attention_mask_landmarks_val = return_ids_mask(encoded_corpus_landmarks_val)
    input_ids_landmarks_test, attention_mask_landmarks_test = return_ids_mask(encoded_corpus_landmarks_test)


    # create dataloaders text and landmarks
    batch_size=512
    train_dataloader = create_dataloaders_multimodal(input_ids_text_train, attention_mask_text_train, text_train_labels, input_ids_landmarks_train, attention_mask_landmarks_train, landmarks_train_labels, batch_size)
    val_dataloader = create_dataloaders_multimodal(input_ids_text_val, attention_mask_text_val, text_val_labels, input_ids_landmarks_val, attention_mask_landmarks_val, landmarks_val_labels, batch_size)
    test_dataloader = create_dataloaders_multimodal(input_ids_text_test, attention_mask_text_test, text_test_labels, input_ids_landmarks_test, attention_mask_landmarks_test, landmarks_test_labels, batch_size, shuffle=False)

    # define the model
    model = CrossBERT("dccuchile/bert-base-spanish-wwm-cased", model_dir_landmarks, 256, 4, 0.2)
    optimizer = AdamW(model.parameters())
    loss_function = nn.MSELoss()

    # set GPU
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logger.debug("Using GPU.")
    else:
        logger.debug("No GPU available, using the CPU instead.")
        device = torch.device("cpu")  
    model.to(device)

    # train and evaluate model
    epochs = 50 # epochs for fine-tunning the new model
    # initialize lists to store the loss and accuracy values
    train_losses = []
    eval_losses = []
    for epoch in range(epochs):
        model, train_loss = train(model, optimizer, loss_function, train_dataloader, device)
        model, eval_loss = evaluate(model, val_dataloader, loss_function, device)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        # print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Eval Loss {eval_loss:.4f}')

    # plot the loss values
    plt.plot(train_losses, color="g")
    plt.plot(eval_losses, color="b")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    output = predict(model, test_dataloader, device)
    
    #  unbatch test labels
    ground_truth = []
    for batch in test_dataloader:
        _, _, labels, _, _, _ = tuple(batch)
        for label in labels:
            ground_truth.append(label)
    for i in range(len(ground_truth)):
        ground_truth[i] = ground_truth[i].tolist()
    
    # CHECK THIS BEFORE RUNNING
    modality = "landmarks"
    vocab_size=1024
    
    mse = mean_squared_error(ground_truth, output)
    ocean_mse = mean_squared_error(ground_truth, output, multioutput="raw_values")
    logger.debug(f"MSE: {mse}")
    logger.debug(f"OCEAN MSE: {ocean_mse}")
    from udiva import UDIVA
    from definitions import *
    metadata = UDIVA.read_metadata("parts", "test")
    parts_ids = metadata[:,0]
    ground_truth = metadata[:,6:]
        
    file_dir = os.path.join("experiments", "multimodal", "data", f"{modality}_test_ids_{vocab_size}.pkl")
    with open(file_dir, "rb") as reader:
        test_ids = pickle.load(reader)
    aggregated_results = []
    for part_id in parts_ids:
        aggregated_results.append(aggregate_results(part_id, test_ids, output))
    aggregated_results = np.asarray(aggregated_results)
    logger.debug(f"Aggregated results: {aggregated_results}")
    OCEAN_LETTERS = ['O', 'C', 'E', 'A', 'N']
    mse = mean_squared_error(ground_truth, aggregated_results, multioutput='raw_values')
    amse = mean_squared_error(ground_truth, aggregated_results)

    logger.debug(f"Results with aggregation")
    for i in range(5):
        logger.debug(f"MSE {OCEAN_LETTERS[i]}: {mse[i]}")
    logger.info(f"AMSE: {amse}")
    len(train_dataloader)