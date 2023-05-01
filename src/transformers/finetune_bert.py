"""
python finetune_bert.py --vocab_size=1024 --log_mode=info --seq_length=128 --modality=landmarks --experiment_id=UL01   --hidden_size=128 --num_tune_epochs=10 --freeze_bert=False --batch_size=32
"""

import os
import pickle
import time
import argparse
import logging
import json
import operator
import random

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BertTokenizer, LineByLineTextDataset, BertConfig, BertForMaskedLM, BertModel

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim import AdamW

from udiva import UDIVA
from definitions import *

# setup logger
logging.basicConfig()
logger = logging.getLogger(__name__)

class BERTRegressor(nn.Module):    
    def __init__(self, drop_rate=0.2, freeze_BERT=False, _model_path=None, _config_path=None, hidden_size=128):
        super(BERTRegressor, self).__init__()
        D_in, D_out = hidden_size, 5
        self.bert = BertModel.from_pretrained(_model_path, config=_config_path, local_files_only=True) 
        # self.regressor = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(D_in, D_out))
        self.regressor = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(drop_rate), nn.Linear(D_in, D_out))
        if freeze_BERT:
            for param in self.bert.parameters():
                param.requires_grad = False
    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1] # outputs[1] ??? why?? CLS??
        outputs = self.regressor(class_label_output)
        return outputs

def read_experiment_setup():
    experiment_details = {} # store experiment details in a dict
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vocab_size', type=int, default=100, help="An integer that determines the vocab size")
    argparser.add_argument("--log_mode", type=str, default="info")
    argparser.add_argument("--seq_length", type=int, default=8)
    argparser.add_argument("--modality", type=str, default="landmarks")
    argparser.add_argument("--experiment_id", type=str, default=0, help="The experiment identifier")
    argparser.add_argument("--hidden_size", type=int, default=128, help="The hidden layers size")
    argparser.add_argument("--num_tune_epochs", type=int, default=100, help="The number of fine tunning epochs")
    argparser.add_argument("--freeze_bert", type=str, default="False", help="The number of fine tunning epochs")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--gpu", type=str, default="7")
    args = argparser.parse_args()
    experiment_details["vocab_size"] = args.vocab_size
    experiment_details["logging_level"] = args.log_mode
    experiment_details["seq_length"] = args.seq_length
    experiment_details["experiment_id"] = args.experiment_id
    experiment_details["hidden_size"] = args.hidden_size
    experiment_details["num_tune_epochs"] = args.num_tune_epochs
    experiment_details["modality"] = args.modality
    experiment_details["batch_size"] = args.batch_size
    experiment_details["gpu"] = args.gpu
    if args.freeze_bert == "False":
        experiment_details["freeze_bert"] = False
    else: experiment_details["freeze_bert"] = True
    return experiment_details

def setup_directories():
    # setup directories
    cwd = os.getcwd()
    experiments_dir = os.path.join(cwd, "experiments")
    data_dir = os.path.join(experiments_dir, "data") # set data directory
    current_exp_dir = os.path.join(experiments_dir, experiment_details["experiment_id"]) # set experiment dir
    #model_dir = os.path.join(current_exp_dir, "model") # set model directory
    #log_dir = os.path.join(current_exp_dir, "logs") # set logger dir
    if not os.path.exists(current_exp_dir):
        os.makedirs(current_exp_dir)
        logger.info(f"The experiment directories have been created successfully.")
    else:
        logger.info(f"The experiment directories already exists.")
    return current_exp_dir, data_dir

def print_logs(log_history, experiment_dir, experiment_details):
    train_loss = []
    epoch_ = []
    eval_loss = []
    for log_ in log_history:
        if "loss" in log_:
            train_loss.append(log_["loss"])
            epoch_.append(log_["epoch"])
        elif "eval_loss" in log_:
            eval_loss.append(log_["eval_loss"])
    for i in range(len(train_loss)):
        logger.info(f"Epoch: {epoch_[i]}, training loss: {train_loss[i]}, eval_loss: {eval_loss[i]}")
    # save the logs for future plotting (maybe make the plot here and get a png at the end of execution???)
    with open(os.path.join(experiment_dir, f'bert_log_history_{experiment_details["vocab_size"]}.pkl'), 'wb') as writer:
        pickle.dump(log_history, writer)
    return train_loss, eval_loss

def plot_and_save(train_loss, eval_loss, amse_list, experiment_details):
    #plt.xlim(0, 1000)
    plt.ylim(0, 1.3)
    xnew = np.linspace(start=0, stop=experiment_details['num_tune_epochs'], num=len(train_loss)) 
    plt.plot(xnew, train_loss, label='train loss')
    xnew = np.linspace(start=0, stop=experiment_details['num_tune_epochs'], num=len(eval_loss)) 
    plt.plot(xnew, eval_loss, label='eval loss')
    xnew = np.linspace(start=0, stop=experiment_details['num_tune_epochs'], num=len(amse_list)) 
    plt.plot(xnew, amse_list, label='amse')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'BERT hidden_size={experiment_details["hidden_size"]}, vocab_size={experiment_details["vocab_size"]}')
    plt.legend()
    # save the figure
    # file_dir = os.path.join(experiment_dir, f'bert_loss_n_heads={experiment_details["num_attention_heads"]}, n_hidden_layers={experiment_details["num_hidden_layers"]}, hidden_size={experiment_details["hidden_size"]}, vocab_size={experiment_details["vocab_size"]}.png')
    file_dir = os.path.join(experiment_dir, f'finetune_plot.png')
    plt.savefig(file_dir)

def load_data_per_split(experiment_details, split, type):
    cwd = os.getcwd()
    if experiment_details['vocab_size'] != 0:
        file_dir = os.path.join(cwd, "experiments", "data", f"{experiment_details['modality']}_{split}_{type}_{experiment_details['vocab_size']}.pkl")
        logger.debug(f"Reading from dir: {file_dir}")
    else: 
        file_dir = os.path.join(cwd, "experiments", "data", f"{experiment_details['modality']}_{split}_{type}.pkl")
        logger.debug(f"Reading from dir: {file_dir}")
    with open(file_dir, "rb") as reader:
        data = pickle.load(reader)
    return data

def load_dataset(experiment_details, split, type):
    if split == "all":
        data_train = load_data_per_split(experiment_details, "train", type)
        data_val = load_data_per_split(experiment_details, "val", type)
        data_test = load_data_per_split(experiment_details, "test", type)
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

def create_dataloaders(inputs, masks, labels, batch_size, shuffle=True):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train(model, optimizer, loss_function, train_dataloader, device):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader): 
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        model.zero_grad()
        outputs = model(batch_inputs, batch_masks)           
        loss = loss_function(outputs.squeeze(), batch_labels.squeeze())
        # print(f"Loss in batch {step} = {loss}")
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
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(), batch_labels.squeeze())
            total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(eval_dataloader)
    return model, avg_valid_loss

def predict(model, dataloader, device):
    model.eval() # sets the module in evaluation mode
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, batch_masks).view(-1,5).tolist()
    return output

def aggregate_results(part_id, test_ids, results):
    """
    Arguments:
    part_id: The current participant ID
    test_ids: ID's from all the samples given to the network
    results: Results returned by the model
    """
    # match the part_id with the test_ids and make a list of indices
    # print(f"Part ID: {part_id}, type: {type(part_id)}")
    # print(f"Test IDs: {test_ids}")
    test_ids = np.array(test_ids)
    indices = np.where(test_ids==part_id)
    indices = indices[0] # use the singleton
    # print(indices)
    # print(f"Indices: {indices}")
    part_results = np.take(results, indices, axis=0)
    # print(f"Part results: {part_results}")
    mean = np.mean(part_results, axis=0)
    # print(f"Mean: {mean}")
    return mean

def preprocess_samples_labels(samples, labels, m):
    new_samples = [] # initialize the new list of lists
    new_labels = []

    for j, sentence in enumerate(samples): # loop through each list in the list of lists
        words = sentence.split() # split the string into a list of words
        for i in range(0, len(words), m): # loop through the words in increments of m
            new_string = ' '.join(words[i:i+m]) # join the m words into a new string
            new_samples.append(new_string) # append the new string to the new list
            new_labels.append(labels[j])
    return new_samples, new_labels

def unbatch_labels(dataloader, ):
    ground_truth = []
    for batch in dataloader:
        _, _, labels = tuple(batch)
        for label in labels:
            ground_truth.append(label)

    for i in range(len(ground_truth)):
        ground_truth[i] = ground_truth[i].tolist() # change OCEAN values to list
    return ground_truth

def compute_aggregated_results(parts_ids, test_ids, output):
    # compute aggregated results per participant
    aggregated_results = []
    for part_id in parts_ids:
        aggregated_results.append(aggregate_results(part_id, test_ids, output))
    aggregated_results = np.asarray(aggregated_results)
    return aggregated_results

if __name__ == "__main__":
    # get the start time
    program_start_time = time.time()
    # read cmd args
    experiment_details = read_experiment_setup()
    # choose GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=experiment_details["gpu"]
    # setup dirs
    experiment_dir, data_dir = setup_directories()
    # configure logger 
    log_file_path = os.path.join(experiment_dir, f"{experiment_details['experiment_id']}_finetune.log")
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)
    if experiment_details["logging_level"] == "info":    
        logger.setLevel(logging.INFO)
    elif experiment_details["logging_level"] == "debug":
        logger.setLevel(logging.DEBUG)
    # print experiment details
    logger.info(f"Beginning experiment with ID: {experiment_details['experiment_id']}")
    logger.info(f"Experiment details: {json.dumps(experiment_details, indent=4)}")
    # fine-tune BERT
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(experiment_dir)
    # load samples and labels
    samples_train, samples_val, samples_test = load_dataset(experiment_details, split="all", type="samples")
    labels_train, labels_val, labels_test = load_dataset(experiment_details, split="all", type="labels")
    # preprocess dataset
    samples_train, labels_train = preprocess_samples_labels(samples_train, labels_train, experiment_details["seq_length"])
    samples_val, labels_val = preprocess_samples_labels(samples_val, labels_val, experiment_details["seq_length"])
    samples_test, labels_test = preprocess_samples_labels(samples_test, labels_test, experiment_details["seq_length"])
    # encode the corpus
    encoded_corpus_train = encode_corpus(data=samples_train, tokenizer=tokenizer)
    encoded_corpus_val = encode_corpus(data=samples_val, tokenizer=tokenizer)
    encoded_corpus_test = encode_corpus(data=samples_test, tokenizer=tokenizer)
    # get ids and masks
    input_ids_train, attention_mask_train = return_ids_mask(encoded_corpus_train)
    input_ids_val, attention_mask_val = return_ids_mask(encoded_corpus_val)
    input_ids_test, attention_mask_test = return_ids_mask(encoded_corpus_test)
    # create dataloaders
    batch_size = experiment_details["batch_size"]
    train_dataloader = create_dataloaders(input_ids_train, attention_mask_train, labels_train, batch_size)
    test_dataloader = create_dataloaders(input_ids_test, attention_mask_test, labels_test, batch_size, shuffle=False)
    val_dataloader = create_dataloaders(input_ids_val, attention_mask_val, labels_val, batch_size)
    # define model
    model = BERTRegressor(drop_rate=0.2, freeze_BERT=experiment_details['freeze_bert'], _model_path=experiment_dir, _config_path=experiment_dir, hidden_size=experiment_details['hidden_size'])
    optimizer = AdamW(model.parameters())
    loss_function = nn.MSELoss()
    # move model to GPU
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logger.info("Using GPU.")
    else:
        logger.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")  
    model.to(device)
    # train and evaluate model
    epochs = experiment_details['num_tune_epochs'] # epochs for fine-tunning the new model
    # load metadata
    metadata = UDIVA.read_metadata("parts", "test")
    parts_ids = metadata[:,0]
    ground_truth = metadata[:,6:]

    # load participant ids from test set
    file_dir = os.path.join("experiments", "data", f"{experiment_details['modality']}_test_ids_{experiment_details['vocab_size']}.pkl")
    with open(file_dir, "rb") as reader:
        test_ids = pickle.load(reader)

    # initialize lists to store the loss and accuracy values
    train_losses = []
    eval_losses = []
    amse_list = []
    OCEAN_LETTERS = ['O', 'C', 'E', 'A', 'N']
    
    for epoch in range(epochs):
        model, train_loss = train(model, optimizer, loss_function, train_dataloader, device)
        model, eval_loss = evaluate(model, val_dataloader, loss_function, device)
        # compute amse with the model after this epoch
        output = predict(model, test_dataloader, device)
        aggregated_results = compute_aggregated_results(parts_ids, test_ids, output)
        mse = mean_squared_error(ground_truth, aggregated_results, multioutput='raw_values')
        amse = mean_squared_error(ground_truth, aggregated_results)
        # logger.info(f"Results after aggregation")
        # for i in range(5):
        #     logger.info(f"MSE {OCEAN_LETTERS[i]}: {mse[i]}")
        # logger.info(f"AMSE: {amse}")
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        amse_list.append(amse)

        logger.info(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Eval Loss {eval_loss:.4f}')
            
    plot_and_save(train_losses, eval_losses, amse_list, experiment_details)

    output = predict(model, test_dataloader, device)
    aggregated_results = compute_aggregated_results(parts_ids, test_ids, output)
    logger.info(f"Aggregated results: {aggregated_results}")
    mse = mean_squared_error(ground_truth, aggregated_results, multioutput='raw_values')
    amse = mean_squared_error(ground_truth, aggregated_results)
    logger.info(f"Results after aggregation")
    for i in range(5):
        logger.info(f"MSE {OCEAN_LETTERS[i]}: {mse[i]}")
    logger.info(f"AMSE: {amse}")

    # get the execution time
    elapsed_time = time.time() - program_start_time
    logger.info(f'Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')