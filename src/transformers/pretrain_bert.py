"""
python pretrain_bert.py --vocab_size=1024 --modality=landmarks --experiment_id=UL01 --log_mode=info --seq_length=128 --hidden_size=128 --num_hidden_layers=2 --num_attention_heads=2 --num_train_epochs=10
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

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BertTokenizer, LineByLineTextDataset, BertConfig, BertForMaskedLM

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# choose GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# setup logger
logging.basicConfig()
logger = logging.getLogger(__name__)

def read_experiment_setup():
    experiment_details = {} # store experiment details in a dict
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vocab_size', type=int, default=100, help="An integer that determines the vocab size")
    argparser.add_argument("--log_mode", type=str, default="info")
    argparser.add_argument("--seq_length", type=int, default=8)
    argparser.add_argument("--modality", type=str, default="landmarks")
    argparser.add_argument("--experiment_id", type=str, default=0, help="The experiment identifier")
    argparser.add_argument("--hidden_size", type=int, default=128, help="The hidden layers size")
    argparser.add_argument("--num_hidden_layers", type=int, default=2, help="The number of hidden layers (encoders)")
    argparser.add_argument("--num_attention_heads", type=int, default=2, help="The number of attention heads")
    argparser.add_argument("--num_train_epochs", type=int, default=100, help="The number of training epochs")
    argparser.add_argument("--freeze_bert", type=str, default="False", help="The number of fine tunning epochs")
    argparser.add_argument("--gpu", type=str, default="7")
    args = argparser.parse_args()
    experiment_details["vocab_size"] = args.vocab_size
    experiment_details["logging_level"] = args.log_mode
    experiment_details["seq_length"] = args.seq_length
    experiment_details["experiment_id"] = args.experiment_id
    experiment_details["hidden_size"] = args.hidden_size
    experiment_details["num_hidden_layers"] = args.num_hidden_layers
    experiment_details["num_attention_heads"] = args.num_attention_heads
    experiment_details["num_train_epochs"] = args.num_train_epochs
    experiment_details["modality"] = args.modality
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
    current_exp_dir = os.path.join(experiments_dir, f'{experiment_details["experiment_id"]}') # set experiment dir
    #model_dir = os.path.join(current_exp_dir, "model") # set model directory
    #log_dir = os.path.join(current_exp_dir, "logs") # set logger dir
    if not os.path.exists(current_exp_dir):
        os.makedirs(current_exp_dir)
        logger.info(f"The experiment directories have been created successfully.")
    else:
        logger.info(f"The experiment directories already exists.")
    return current_exp_dir, data_dir

def cut_corpus(file_path, seq_length):
    # crop the dataset
    # delete words less than 8
    n = seq_length # set the minimum number of words per line
    output_file_path = f"{file_path[:-4]}_cut.txt"
    with open(file_path, 'r') as f, open(output_file_path, 'w') as out_file:
        for line in f:
            if len(line.split()) >= n: # check if the line has at least n words
                out_file.write(line) # write the line to the output file
    return output_file_path

def resize_corpus(file_path, seq_length):
    # resize the corpus lines
    output_file_path = f"{file_path[:-4]}_resized.txt"
    m = seq_length # maximum number of words in each output line
    with open(file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        for line in input_file:
            words = line.split()
            for i in range(0, len(words), m):
                output_line = " ".join(words[i:i+m])
                output_file.write(output_line + "\n")
    return output_file_path

def shuffle_corpus(file_path):
    # load and shuffle training corpus
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # shuffle lines
    random.shuffle(lines)
    # write shuffled lines back to input file
    output_file_path = f"{file_path[:-4]}_shuffled.txt"
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_file_path

def preprocess_corpus(file_path, seq_length):
    logger.info(file_path)
    file_path = shuffle_corpus(file_path)
    file_path = resize_corpus(file_path, seq_length)
    file_path = cut_corpus(file_path, seq_length)
    return file_path

def load_corpus(experiment_details, split):
    file_path = os.path.join(data_dir, f"{experiment_details['modality']}_{split}_corpus_{experiment_details['vocab_size']}.txt")
    seq_length = experiment_details['seq_length']
    file_path = preprocess_corpus(file_path, seq_length)
    # load dataset
    dataset = LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = seq_length # 128 was default
    )
    logger.info(f"Loaded dataset with {len(dataset)} lines")
    return dataset

def train_tokenizer(experiment_dir, data_dir, experiment_details):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    file_path = os.path.join(data_dir, f"{experiment_details['modality']}_train_corpus_{experiment_details['vocab_size']}.txt")
    logger.info(file_path) 
    files = [file_path]
    pre_tokenizer = Whitespace()
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $0 [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer.train(files, trainer)
    vocab = tokenizer.get_vocab()
    sorted_vocab = dict(sorted(vocab.items(), key=operator.itemgetter(1)))
    """
    logger.info("Tokenizer vocabulary")
    for word in sorted_vocab:
        logger.info(word)
    """
    file_path = os.path.join(experiment_dir, "vocab.txt")
    with open(file_path, "w") as writer:
        for word in sorted_vocab:
            writer.write(f"{word}\n")
    logger.info(f"Trained tokenizer successfully, saved in {file_path}")

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
    # with open(os.path.join(experiment_dir, f'bert_log_history_{experiment_details["vocab_size"]}.pkl'), 'wb') as writer:
        # pickle.dump(log_history, writer)
    return train_loss, eval_loss

def plot_and_save(train_loss, eval_loss, experiment_details):
    #plt.xlim(0, 1000)
    plt.ylim(0, 10)
    xnew = np.linspace(start=0, stop=experiment_details['num_train_epochs'], num=len(train_loss)) 
    plt.plot(xnew, train_loss, label='train loss')
    xnew = np.linspace(start=0, stop=experiment_details['num_train_epochs'], num=len(eval_loss)) 
    plt.plot(xnew, eval_loss, label='eval loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'BERT n_heads={experiment_details["num_attention_heads"]}, n_hidden_layers={experiment_details["num_hidden_layers"]}, hidden_size={experiment_details["hidden_size"]}, vocab_size={experiment_details["vocab_size"]}')
    plt.legend()
    # save the figure
    # file_dir = os.path.join(experiment_dir, f'bert_loss_n_heads={experiment_details["num_attention_heads"]}, n_hidden_layers={experiment_details["num_hidden_layers"]}, hidden_size={experiment_details["hidden_size"]}, vocab_size={experiment_details["vocab_size"]}.png')
    file_dir = os.path.join(experiment_dir, f'pretrain_plot.png')
    plt.savefig(file_dir)

if __name__ == "__main__":
    # get the start time
    program_start_time = time.time()
    # read cmd args
    experiment_details = read_experiment_setup()
    experiment_dir, data_dir = setup_directories()
    # choose GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=experiment_details["gpu"]
    # configure logger 
    log_file_path = os.path.join(experiment_dir, f"{experiment_details['experiment_id']}_pretrain.log")
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)
    if experiment_details["logging_level"] == "info":    
        logger.setLevel(logging.INFO)
    elif experiment_details["logging_level"] == "debug":
        logger.setLevel(logging.DEBUG)
    # print experiment details
    logger.info(f"Beginning experiment with ID: {experiment_details['experiment_id']}")
    logger.info(f"Experiment details: {json.dumps(experiment_details, indent=4)}")
    # train tokenizer
    train_tokenizer(experiment_dir, data_dir, experiment_details)
    # load tokenizer trained in train_tokenizer.ipynb
    file_path = os.path.join(experiment_dir, "vocab.txt")
    logger.info("Loading tokenizer from pretrained")
    tokenizer = BertTokenizer.from_pretrained(file_path)
    logger.info(tokenizer)
    # load corpus files
    train_dataset = load_corpus(experiment_details, "train")
    val_dataset = load_corpus(experiment_details, "val")
    # configure bert
    config = BertConfig(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=experiment_details['hidden_size'], 
        num_hidden_layers=experiment_details['num_hidden_layers'], 
        num_attention_heads=experiment_details['num_attention_heads'], 
        max_position_embeddings=512
    )
    # set pretraining task
    model = BertForMaskedLM(config)
    logger.info(f'No of parameters: {model.num_parameters()}')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    # set training ars
    training_args = TrainingArguments(
        #output_dir='./',
        output_dir=experiment_dir,
        overwrite_output_dir=True,
        num_train_epochs=experiment_details['num_train_epochs'],
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        do_eval=True,
        evaluation_strategy="steps"
    )
    # set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
        #prediction_loss_only=True,
    )
    logger.info("Starting training process")
    trainer.train()
    logger.info("Training complete")
    log_history = trainer.state.log_history
    train_loss, eval_loss = print_logs(log_history, experiment_dir, experiment_details)
    plot_and_save(train_loss, eval_loss, experiment_details)
    trainer.save_model(experiment_dir)
    logger.info("Model saved to disk")
    # get the execution time
    elapsed_time = time.time() - program_start_time
    logger.info(f'Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')