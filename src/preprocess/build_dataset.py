from udiva import UDIVA
import numpy as np
from definitions import *
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import sys
import logging
import argparse
import time

"""
DESCRIPTION:

TODO: 
1. Print the path of every saved archive using the logger
"""

logging.basicConfig()
logger = logging.getLogger(__name__)

def find_most_similar(vector, vocab):
    # find most similar using cosine similarity sklearn
    X = [value for key, value in vocab.items()]
    keys = [key for key, value in vocab.items()]
    X = np.asarray(X)
    y = [vector]
    y = np.asarray(y)
    sim = cosine_similarity(X,y)
    index_min = max(range(len(sim)), key=sim.__getitem__) # returns index of max element 
    ret_key = keys[index_min]
    return ret_key

if __name__ == "__main__":
    # get the start time
    program_start_time = time.time()
    # read cmd arguments with argparser
    argparser = argparse.ArgumentParser(description='Read arguments')
    argparser.add_argument('--vocab_size', type=int, default=100, help="An integer that determines the vocab size")
    argparser.add_argument('--experiment_name', type=str, default="delete", help='The experiment name')
    argparser.add_argument('--split', type=str, default="test", help='The split used to build the dataset')
    argparser.add_argument("--log_mode", type=str, default="info", help="Logging mode")
    argparser.add_argument("--modality", type=str, default="landmarks", help="The modality to work with")
    argparser.add_argument("--word_length", type=int, default=5, help="The word length for the vocab")
    args = argparser.parse_args()
    vocab_size = args.vocab_size
    experiment_name = args.experiment_name
    split = args.split
    logging_level = args.log_mode
    modality = args.modality
    word_length = args.word_length
    # create directory for experiment
    cwd = os.getcwd()
    experiment_dir = os.path.join(cwd, 'experiments', modality, experiment_name)
    # set logging file path
    log_dir = os.path.join(experiment_dir, 'logs', f'build_dataset_{split}_{vocab_size}.log')
    # configure logger
    file_handler = logging.FileHandler(log_dir)
    logger.addHandler(file_handler)
    if logging_level == "info":    
        logger.setLevel(logging.INFO)
    elif logging_level == "debug":
        logger.setLevel(logging.DEBUG)
    logger.info(f"Experiment details: experiment_name={experiment_name}, vocab_size={vocab_size}, modality={modality}, word_length={word_length}")
    # set data directory
    data_dir = os.path.join(experiment_dir, 'data')
    # start to build vocab
    parts_metadata = UDIVA.read_metadata('parts', split)
    # load data
    file_dir = os.path.join(data_dir, f'{modality}_{split}_all_tasks.pkl')
    with open(file_dir, "rb") as reader:
        data = pickle.load(reader)
    # load vocab and scaler
    file_dir = os.path.join(data_dir, f'{modality}_vocab_{vocab_size}.pkl')
    with open(file_dir, "rb") as reader:
        vocab = pickle.load(reader)
    file_dir = os.path.join(data_dir, f'scaler_{vocab_size}.pkl')
    with open(file_dir, "rb") as reader:
        scaler = pickle.load(reader)
    samples = []
    labels = []
    ids = []
    for part in data:
        logger.info(f"Computing data from participant {part}")
        for session in data[part]:
            for task in data[part][session]:
                sentence = ""
                for utterance in data[part][session][task]:
                    for i in range(len(utterance)):
                        vector = utterance[i].reshape(1,-1)
                        vector = scaler.transform(vector)
                        sentence = sentence + " " + find_most_similar(vector[0], vocab)
                idx = np.where(parts_metadata[:,0] == part)[0][0] 
                labels.append(parts_metadata[idx][6::])
                samples.append(sentence)
                ids.append(part)
    if split == "train" or split == "val":
        file_dir = os.path.join(data_dir, f'{modality}_{split}_corpus_{vocab_size}.txt')
        with open(file_dir, "w") as writer:
            for sentence in samples:
                writer.write(f"{sentence}\n")
    # save samples to disk
    file_dir = os.path.join(data_dir, f'{modality}_{split}_samples_{vocab_size}.pkl')
    with open(file_dir, "wb") as writer:
        pickle.dump(samples, writer)
    file_dir = os.path.join(data_dir, f'{modality}_{split}_labels_{vocab_size}.pkl')
    with open(file_dir, "wb") as writer:
        pickle.dump(labels, writer)
    file_dir = os.path.join(data_dir, f'{modality}_{split}_ids_{vocab_size}.pkl')
    with open(file_dir, "wb") as writer:
        pickle.dump(ids, writer)
    logger.info(f"Program finished successfully")
    # get the execution time
    elapsed_time = time.time() - program_start_time
    logger.info(f'Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')