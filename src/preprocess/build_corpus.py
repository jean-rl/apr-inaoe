from udiva import UDIVA
import numpy as np
from definitions import *
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import sys

import logging
mylogs = logging.getLogger(__name__)
mylogs.setLevel(logging.INFO)

# find most similar using cosine similarity sklearn
def find_most_similar(vector, vocab):
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
    SPLITS = ["train", "test"]

    cmdargs = str(sys.argv)
    vocab_size = int(sys.argv[1])
    experiment_name = str(sys.argv[2])
    modality = str(sys.argv[3])

    for split in SPLITS:
        # set logging directory
        cwd = os.getcwd()
        dir = os.path.join(cwd, 'experiments', modality, experiment_name)
        log_dir = os.path.join(dir, 'logs', f'build_corpus_{split}_{vocab_size}.log')
        file = logging.FileHandler(log_dir)
        mylogs.addHandler(file)
        # set data directory
        data_dir = os.path.join(dir, 'data')
        mylogs.info(f"Starting program - Build corpus vocab size: {vocab_size}")
        # split = "train"
        # vocab_size = 1000

        word_length = 5
        parts_metadata = UDIVA.read_metadata('parts', split)
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
            mylogs.info(f"Loading data from participant: {part} - with vocab size: {vocab_size}")
            for session in data[part]:
                for task in data[part][session]:
                    for utterance in data[part][session][task]:
                        sentence = ""
                        for i in range(len(utterance)):
                            vector = utterance[i].reshape(1,-1)
                            vector = scaler.transform(vector)
                            sentence = sentence + " " + find_most_similar(vector[0], vocab)
                        idx = np.where(parts_metadata[:,0] == part)[0][0] 
                        labels.append(parts_metadata[idx][6::])
                        samples.append(sentence)
                        ids.append(part)
        if split == "train":
            file_dir = os.path.join(data_dir, f'{modality}_train_corpus_{vocab_size}.txt')
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
        mylogs.info(f"Finished program - Build corpus vocab size: {vocab_size} - split: {split}")