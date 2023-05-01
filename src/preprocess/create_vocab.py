#from udiva import UDIVA
import numpy as np
from definitions import *
import pickle
import random
import string
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import argparse
import logging
import time

"""
The code creates a vocabulary using a clustering method. It receives an NxM matrix where N is the number of vectors and M is the number of features. It then creates c clusters and a fake vocabulary of random words using each cluster.

The code contains the following main routine:

Parses command-line arguments to determine the vocabulary size, experiment name, logging mode, modality, word length, and clustering method.
Configures the logger and sets up the experiment directory and data directory.
Loads the data.
Calls get_number_samples to get the total number of samples.
Calls build_matrix to create a matrix containing all the vectors from the data.
Calls normalize_matrix to normalize the matrix.
Calls either create_clusters_fast or create_clusters depending on the specified clustering method to create the clusters.
Calls clusters_to_words to create a vocabulary of random words based on the cluster centers.
Saves the resulting vocabulary as a Python pickle file.
"""

logging.basicConfig()
logger = logging.getLogger(__name__)

def get_number_samples(data):
    """Given a dictionary containing the data, returns the total number of samples."""
    sum = 0
    for part in data.keys():
        # sum = sum + data[part].shape[0]
        for session in data[part].keys():
            for task in data[part][session].keys():
                for utterance in data[part][session][task]:
                    #logger.info(f"part: {part}, session: {session}, task: {task}, data shape: {len(utterance)}")
                    sum = sum + len(utterance)
    logger.info(f"Total number of samples: {sum}")
    return sum

def build_matrix(landmarks, num_samples, length):
    """Given a dictionary of landmarks, the total number of samples, and the length of each sample, returns a matrix containing all the vectors from the landmarks."""
    matrix = np.zeros((num_samples, length))
    i = 0
    for part in landmarks:
        for session in landmarks[part]:
            for task in landmarks[part][session]:
                for utterance in landmarks[part][session][task]:
                    for vector in utterance:
                        matrix[i] = vector
                        i = i+1
    return matrix

def normalize_matrix(matrix, vocab_size):
    """Given a matrix and the desired vocabulary size, returns a normalized matrix."""
    scaler = MinMaxScaler()
    scaler.fit(matrix)
    matrix = scaler.transform(matrix)
    return matrix, scaler

def create_clusters_fast(matrix, n_clusters_):
    """Given a matrix and the number of clusters, uses the MiniBatchKMeans clustering algorithm to create the clusters and returns the results."""
    minikmeans = MiniBatchKMeans(n_clusters=n_clusters_, random_state=0, n_init="auto").fit(matrix)
    logger.debug(f"Created clusters using MiniBatchKMeans with settings: n_clusters={n_clusters_}, random_state=0, n_init=auto")
    return minikmeans

def create_clusters(matrix, n_clusters_):
    """Given a matrix and the number of clusters, uses the KMeans clustering algorithm to create the clusters and returns the results."""
    kmeans = KMeans(n_clusters=n_clusters_, random_state=0, n_init="auto").fit(matrix)
    logger.debug(f"Created clusters using KMeans with settings: n_clusters={n_clusters_}, random_state=0, n_init=auto")
    return kmeans

def get_random_string(length):
    """Given a length, returns a random string."""
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def clusters_to_words(kmeans, word_length):
    """Given a KMeans object and a word length, creates a vocabulary of random words based on the cluster centers and returns the results."""
    vocab = {}
    for vector in kmeans.cluster_centers_:
        word = get_random_string(word_length)
        vocab[word] = vector
    return vocab

if __name__ == "__main__":
    # get the start time
    program_start_time = time.time()
    # read cmd args
    argparser = argparse.ArgumentParser(description='Read arguments')
    argparser.add_argument('--vocab_size', type=int, default=100, help="An integer that determines the vocab size")
    argparser.add_argument('--experiment_name', type=str, default="none", help='The experiment name')
    argparser.add_argument("--log_mode", type=str, default="info")
    argparser.add_argument("--modality", type=str, default="landmarks", help="The modality to work with")
    argparser.add_argument("--word_length", type=int, default=5, help="The word length for the vocab")
    argparser.add_argument("--clustering", type=str, default="fast", help="The clustering method")
    args = argparser.parse_args()
    vocab_size = args.vocab_size
    experiment_name = args.experiment_name
    logging_level = args.log_mode
    modality = args.modality
    word_length = args.word_length
    clustering_method = args.clustering
    # set experiment directory
    cwd = os.getcwd()
    experiment_dir = os.path.join(cwd, 'experiments', experiment_name)
    # set logging file path
    log_dir = os.path.join(experiment_dir, 'logs', f'create_vocab_size_{vocab_size}.log')
    # configure logger
    file_handler = logging.FileHandler(log_dir)
    logger.addHandler(file_handler)
    if logging_level == "info":    
        logger.setLevel(logging.INFO)
    elif logging_level == "debug":
        logger.setLevel(logging.DEBUG)
    logger.info(f"Experiment details: experiment_name={experiment_name}, vocab_size={vocab_size}, modality={modality}, word_length={word_length}, clustering_method={clustering_method}")
    # set data directory
    data_dir = os.path.join(experiment_dir, 'data')
    # load data
    file_dir = os.path.join(data_dir, f'{modality}_train.pkl')
    with open(file_dir, "rb") as reader:
        data = pickle.load(reader)
    # read an example shape from the data
    data_shape = data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]][list(data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]].keys())[0]][0].shape
    num_samples = get_number_samples(data)
    logger.debug(f"Data loaded with shape: {data_shape} and {num_samples} samples")
    logger.info("Building matrix")
    matrix = build_matrix(data, num_samples, data_shape[1])
    logger.info(f"Matrix shape: {matrix.shape}")
    matrix, scaler = normalize_matrix(matrix, vocab_size)
    # save scaler
    file_dir = os.path.join(data_dir, f"scaler_{vocab_size}.pkl")
    with open(file_dir, "wb") as writer:
        pickle.dump(scaler, writer)
    logger.info(f"Matrix normalized and scaler saved")
    logger.info(f"Creating clusters using the {clustering_method} method")
    if clustering_method == "fast":
        kmeans = create_clusters_fast(matrix, vocab_size) # you can change it to create_clusters_fast
    elif clustering_method == "slow":
        kmeans = create_clusters(matrix, vocab_size)
    logger.info(f"Clusters created: {kmeans.cluster_centers_.shape}.")
    vocab = clusters_to_words(kmeans, word_length)
    logger.info(f"Vocabulary created with {len(vocab)} words.")
    file_dir = os.path.join(data_dir, f"{modality}_vocab_{vocab_size}.pkl")
    with open(file_dir, "wb") as writer:
        pickle.dump(vocab, writer)
    logger.info(f"Vocabulary created and saved")
    # get the execution time
    elapsed_time = time.time() - program_start_time
    logger.info(f'Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')