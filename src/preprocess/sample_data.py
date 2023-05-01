from udiva import UDIVA
import numpy as np
from definitions import *
import math
import pickle
import logging
import argparse
import time 
import os 
# landmark related libraries
import h5py
# audio related libraries
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
# load the VGGish model
# vgg, sess = UDIVA.load_audio_model()
# set GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
# set logger
logging.basicConfig() 
logger = logging.getLogger(__name__)

model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def get_number_of_samples(data):    
    """Returns the total number of vectors in the dataset."""
    sum = 0
    for part in data.keys():
        for session in data[part].keys():
            for task in data[part][session].keys():
                for utterance in data[part][session][task]:
                    sum = sum + len(utterance)
    return sum

def sample_face_frames(part_id, session_id, task, split, sr):
    """Samples face frames from the specified task at a sampling rate of sr and returns them as an array."""
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
    face_frames = np.empty((0, 204)) # stores 68 face 3D coordinates
    with h5py.File(file_name, 'r') as reader:
        frames = list(reader.keys())
        logger.info(f"Part: {part_id}, session: {session_id}, task: {task}, total frames in the session: {len(frames)}")       
        # read data
        for i in range(0, len(frames), sr):
            try: # since the compute of the frames could take some index out of range
                data = reader[frames[i]] 
            except IndexError:
                break
            if not data.keys(): # empty keys indicate the algorithm could not compute frames
                continue
            if data.attrs['valid'] == False: # frames in which the algorithm has a high error were marked as "invalid"
                logger.info('False frame')
            face = data['face']
            if 'landmarks' in face.keys() and face.attrs['valid'] == True:
                flattened = np.ndarray.flatten(np.array(list(face['landmarks'])))
                face_frames = np.vstack((face_frames, flattened))
    return face_frames

def sample_face_frames_time(part_id, session_id, task, split, sr):
    """Samples at utterance level each sr frames. """
    times = UDIVA.get_times_for_part(part_id, session_id, task, split)
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
    utterances = []
    with h5py.File(file_name, 'r') as reader:
        frames = list(reader.keys())
        logger.info(f"Part: {part_id}, session: {session_id}, task: {task}, total frames in the session: {len(frames)}")       
        # read data
        for utterance in times:
            # computing times -> seconds to frames
            start_time = float(utterance[1])
            end_time = float(utterance[2])
            logger.debug(f"Start time: {start_time}, end time: {end_time}")
            total_time = end_time - start_time
            logger.debug(f"Total time {total_time}")
            start_frame = int(FPS * start_time)
            end_frame = int(FPS * end_time)
            # total_frames = int(FPS * total_time)
            face_frames = np.empty((0, 204)) # stores 68 face 3D coordinates
            for i in range(start_frame, end_frame+1, sr):
                try: # since the compute of the frames could take some index out of range
                    data = reader[frames[i]] 
                except IndexError:
                    break
                if not data.keys(): # empty keys indicate the algorithm could not compute frames
                    continue
                if data.attrs['valid'] == False: # frames in which the algorithm has a high error were marked as "invalid"
                    logger.debug('False frame')
                face = data['face']
                if 'landmarks' in face.keys() and face.attrs['valid'] == True:
                    flattened = np.ndarray.flatten(np.array(list(face['landmarks'])))
                    face_frames = np.vstack((face_frames, flattened))
            utterances.append(face_frames)
    return utterances

def sample_face_frames_time_diff(part_id, session_id, task, split, sr):
    """Samples face frames from the specified task at the utterance level and a 
    sampling rate of sr and computes the difference between frames, 
    returning the difference frames as a list of arrays."""
    times = UDIVA.get_times_for_part(part_id, session_id, task, split)
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
    utterances = []
    with h5py.File(file_name, 'r') as reader:
        frames = list(reader.keys())
        logger.info(f"Part: {part_id}, session: {session_id}, task: {task}, total frames in the session: {len(frames)}")       
        # read data
        for utterance in times:
            # computing times -> seconds to frames
            start_time = float(utterance[1])
            end_time = float(utterance[2])
            # logger.info(f"Start time: {start_time}, end time: {end_time}")
            # total_time = end_time - start_time
            # logger.info(f"Total time {total_time}")
            start_frame = int(FPS * start_time)
            end_frame = int(FPS * end_time)
            # total_frames = int(FPS * total_time)
            face_frames = np.empty((0, 204)) # stores 68 face 3D coordinates
            for i in range(start_frame, end_frame+1, sr): # samples from an utterance
                try: # since the compute of the frames could take some index out of range
                    data_1 = reader[frames[i]] 
                    data_2 = reader[frames[i+1]]
                except IndexError:
                    break
                if (not data_1.keys()) or (not data_2.keys()): # empty keys indicate the algorithm could not compute frames
                    continue
                if (data_1.attrs['valid'] == False) or (data_2.attrs['valid'] == False): # frames in which the algorithm has a high error were marked as "invalid"
                    logger.info('False frame')
                face_1 = data_1['face']
                face_2 = data_2['face']
                if ('landmarks' in face_1.keys() and face_1.attrs['valid'] == True) and ('landmarks' in face_2.keys() and face_2.attrs['valid'] == True):
                    flattened_1 = np.ndarray.flatten(np.array(list(face_1['landmarks'])))
                    flattened_2 = np.ndarray.flatten(np.array(list(face_2['landmarks'])))
                    diff = np.subtract(flattened_1, flattened_2)
                    face_frames = np.vstack((face_frames, diff))
            utterances.append(face_frames)
    return utterances

def sample_audio_utterance(part_id, session_id, task, split):
    """
    Samples audio at utterance level
    """
    logger.debug(f"Part: {part_id}, session: {session_id}, task: {task}")       
    times = UDIVA.get_times_for_part(part_id, session_id, task, split)
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_wav_filename(part_name, session_id, task, split)
    print("File name: ", file_name)
    # load audio clip
    data, sr = librosa.load(file_name) # sr is the sampling rate
    logger.debug(f"Sampling @: {sr} Hz")
    utterances = []
    for utterance in times:
        embeddings = np.empty((0,128), dtype=np.uint8)
        # compute the start and end time of the utterance
        start_time = float(utterance[1])
        end_time = float(utterance[2])
        logger.debug(f"Start time: {start_time} s, end time: {end_time} s")
        total_time = end_time - start_time
        logger.debug(f"Total time: {total_time} s")
        start_index = int(float(start_time*sr))
        end_index = int(float(end_time*sr))
        logger.debug(f"Start index: {start_index}, end index: {end_index}")
        total_samples = end_index - start_index
        logger.debug(f"Total samples in that period of time {total_samples}")
        # take the slice
        audio = data[start_index:end_index]
        # if there's an audio shorter than a second (VGGIsh problem) we pad it with zeros to make it a second
        if end_index - start_index < sr: 
            logger.debug("Length is shorter than expected, padding with zeros...")
            audio = np.concatenate((audio, np.zeros(sr - (end_index - start_index))))
            logger.debug(f"New length of the sample: {len(audio)}")
            # audio = np.pad(audio, (0, sr - (end_index - start_index)), 'constant')
        # compute the embedding using the VGGish model
        # vgg, sess = UDIVA.load_audio_model()
        try:
            embedding = UDIVA.ProcessWithVGGish(vgg, audio, sr, sess)
            embeddings = np.vstack((embeddings, embedding))
            logger.debug("Processed audio successfully with the VGGish network!")
        except:
            logger.debug(f"Error in computing the embedding for part {part_id}, session {session_id}, task {task}")
            continue
            # store the utterance representation
        utterances.append(embeddings)
    return utterances   

def sample_audio_time_vggish(part_id, session_id, task, split, st):
    """
    Samples audio with a given milliseconds window determined by st argument
    """
    logger.info(f"Part: {part_id}, session: {session_id}, task: {task}")       
    times = UDIVA.get_times_for_part(part_id, session_id, task, split)
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_wav_filename(part_name, session_id, task, split)
    print("File name: ", file_name)
    # load audio clip
    data, sr = librosa.load(file_name) # sr is the sampling rate
    # convert the sampling time to samples
    st = int(float(sr * st))
    utterances = []
    for utterance in times:
        embeddings = np.empty((0,128), dtype=np.uint8)
        # compute the start and end time of the utterance
        start_time = float(utterance[1])
        end_time = float(utterance[2])
        logger.debug(f"Start time: {start_time}, end time: {end_time}")
        total_time = end_time - start_time
        logger.debug(f"Total time {total_time}")
        start_index = int(float(start_time*sr))
        end_index = int(float(end_time*sr))
        total_samples = end_index - start_index
        logger.debug(f"Samples in that period of time {total_samples}")
        logger.debug(f"Start frame {start_index}, end frame {end_index}")
        # take the slice
        audio = data[start_index:end_index]
        # compute how many times st fits in the utterance
        num_samples = math.floor(total_samples/st)
        logger.debug(f"Number of samples {num_samples}")

        for i in range(num_samples):
            _start_index = st * i
            _end_index = st * (i+1)
            logger.debug(f"Start index {start_index}, end index {end_index}")
            # take the slice from the audio
            sample = audio[_start_index:_end_index]
            # if there's an audio shorter than a second (VGGIsh problem) we pad it with zeros to make it a second
            if _end_index - _start_index < sr: 
                sample = np.concatenate((sample, np.zeros(sr - (_end_index - _start_index))))
                # audio = np.pad(audio, (0, sr - (end_index - start_index)), 'constant')
            # compute the embedding using the VGGish model
            # vgg, sess = UDIVA.load_audio_model()
            try:
                embedding = UDIVA.ProcessWithVGGish(vgg, sample, sr, sess)
                embeddings = np.vstack((embeddings, embedding))
            except:
                logger.debug(f"Error in computing the embedding for part {part_id}, session {session_id}, task {task}")
                continue
            # store the utterance representation
        utterances.append(embeddings)
    return utterances   

def sample_audio_time_wav2vec(part_id, session_id, task, split, st):
    """
    Samples audio with a given milliseconds window determined by st argument
    """
    logger.debug(f"Part: {part_id}, session: {session_id}, task: {task}")       
    times = UDIVA.get_times_for_part(part_id, session_id, task, split)
    part_name = UDIVA.get_name_for_part(part_id, session_id, split)
    file_name = UDIVA.generate_wav_filename(part_name, session_id, task, split)
    logger.info(f"File name: {file_name}")
    # load audio clip
    data, sr = librosa.load(file_name, sr=16000) # undersample to 16kHz
    # convert the sampling time to samples
    st = int(float(sr * st))
    utterances = []
    for utterance in times:
        embeddings = np.empty((0,1024), dtype=np.uint8)
        # compute the start and end time of the utterance
        start_time = float(utterance[1])
        end_time = float(utterance[2])
        logger.debug(f"Start time: {start_time}, end time: {end_time}")
        total_time = end_time - start_time
        logger.debug(f"Total time {total_time}")
        start_index = int(float(start_time*sr))
        end_index = int(float(end_time*sr))
        total_samples = end_index - start_index
        logger.debug(f"Samples in that period of time {total_samples}")
        logger.debug(f"Start frame {start_index}, end frame {end_index}")
        # take the slice
        audio = data[start_index:end_index]
        # compute how many times st fits in the utterance
        num_samples = math.floor(total_samples/st)
        logger.debug(f"Number of samples: {num_samples}")

        for i in range(num_samples):
            _start_index = st * i
            _end_index = st * (i+1)
            logger.debug(f"Start index {start_index}, end index {end_index}")
            # take the slice from the audio
            sample = audio[_start_index:_end_index]
            # compute embedding
            try:
                input = feature_extractor(sample, return_tensors="pt", sampling_rate=16000)
                with torch.no_grad():
                    output = model(input.input_values)
                embedding = torch.mean(output.last_hidden_state, dim=1)
                embeddings = np.vstack((embeddings, embedding))
            except:
                logger.debug(f"Error in computing the embedding for part {part_id}, session {session_id}, task {task}")
                continue
            # store the utterance representation
        utterances.append(embeddings)
    return utterances   

def sample_data(parts_metadata, modality):
    """Sample process for a given modality."""
    parts_data = {}
    for i, part in enumerate(parts_metadata[:,0]):
        session_data = {}
        for session in parts_metadata[i,1:6]:   
            if math.isnan(session):
                continue
            else:
                task_data = {}
                for task in TASKS:
                    if modality == "landmarks":
                        frames = sample_face_frames_time(part, int(session), task, split, 5) # samples each 5 frames (200ms)
                    elif modality == "audio":
                        frames = sample_audio_time_wav2vec(part, int(session), task, split, 0.2) # samples each 200 ms
                    task_data[task] = frames
                session_data[int(session)] = task_data
        parts_data[part] = session_data
    return parts_data

if __name__ == "__main__":
    # get the start time
    program_start_time = time.time()
    # read cmd arguments with argparser
    argparser = argparse.ArgumentParser(description='Read arguments')
    argparser.add_argument('--split', type=str, default="test", help="The selected split")
    argparser.add_argument('--experiment_name', type=str, default="delete", help='The experiment name')
    argparser.add_argument("--log_mode", type=str, default="info")
    argparser.add_argument("--modality", type=str, help="Current supported modalities are landmarks and audio")
    args = argparser.parse_args()
    split = args.split
    experiment_name = args.experiment_name
    logging_level = args.log_mode
    modality = args.modality
    # set experiment directory
    cwd = os.getcwd()
    experiment_dir = os.path.join(cwd, 'experiments', experiment_name)
    # set logging file path
    log_dir = os.path.join(experiment_dir, 'logs', f'sample_{modality}_{split}.log')
    # setup the logger
    file_handler = logging.FileHandler(log_dir)
    logger.addHandler(file_handler)
    if logging_level == "info":    
        logger.setLevel(logging.INFO)
    elif logging_level == "debug":
        logger.setLevel(logging.DEBUG)
    # start sampling process
    logger.info(f"Starting to sample {modality} for {split} split")
    parts_metadata = UDIVA.read_metadata('parts', split)
    parts_data = sample_data(parts_metadata, modality)
    logger.info(f"Finished sampling {modality}")
    # store data with a given format
    file_name = os.path.join(experiment_dir, 'data', f'{modality}_{split}.pkl')
    with open(file_name, "wb") as writer:
        pickle.dump(parts_data, writer)
    logger.info(f'Saved {get_number_of_samples(parts_data)} samples for the {split} split')
    # get the execution time
    elapsed_time = time.time() - program_start_time
    logger.info(f'Execution time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')