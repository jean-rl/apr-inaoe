# UDIVA v2 - This implementation removes old code from recurrent neural networks.

# testing

import numpy as np
import pandas as pd
import os # building path names
import srt # reading transcript files
import codecs # reading transcript files
import h5py # reading landmark files
from definitions import * # using all UDIVA constants
import sys 
import math # checking for nan
#import librosa # read wav files
#from gensim.models import fasttext # text embeddings
import regex as re # tokenize text
# python.exe -m pip install googletrans==3.1.0a0
#from googletrans import Translator # google translator

"""
The logging library will be used to store logs in files
Still needs modification to specify a new file each time the code runs
"""
import logging
"""
logger = logging.getLogger()
# uncomment if you want to send prints to log
# print = logger.info
sys.stderr.write = logger.error
# sys.stdout.write = logger.info
logging.basicConfig(filename="logs.log", filemode="w", level=logging.DEBUG)
"""
class UDIVA:

    """
    Methods to keep logs
    """
    @staticmethod
    def set_logger_file(file):
        logger = logging.getLogger()
        # uncomment if you want to send prints to log
        print = logger.info
        sys.stderr.write = logger.error
        sys.stdout.write = logger.info 
        logging.basicConfig(filename=file, filemode="w", level=logging.DEBUG)

    @staticmethod
    def set_logger_print(file):
        logger = logging.getLogger()
        # uncomment if you want to send prints to log
        print = logger.info
        sys.stderr.write = logger.error
        sys.stdout.write = logger.info 
        logging.basicConfig(filename=file, filemode="w", level=logging.DEBUG)

    """
    Generic methods used to extract features for every modality
    """

    """
    Reads the metadata files
    """
    @staticmethod
    def read_metadata(type, split):
        logging.info(f"Reading {type} metadata from the {split} split")
        if type == 'sessions':
            file_name = f'{type}_{split}.csv'
            df = pd.read_csv(f'{DATA_DIR}/{split}/metadata/{file_name}')
            metadata = np.column_stack((df['ID'].to_numpy(), df['PART.1'].to_numpy(), 
                                        df['PART.2'].to_numpy(), df['LANGUAGE'].to_numpy()))
        elif type == 'parts':
            if split == 'test' or split == 'val':
                file_name = f'{type}_{split}_unmasked.csv'
            else:
                file_name = f'{type}_{split}.csv'
            df = pd.read_csv(f'{DATA_DIR}/{split}/metadata/{file_name}')
            metadata = np.column_stack((df['ID'].to_numpy().astype(object), df.loc[:,'SESSION1':'SESSION5'].to_numpy().astype(object), 
                                        df.loc[:,'OPENMINDEDNESS_Z':'NEGATIVEEMOTIONALITY_Z'].to_numpy().astype(object)))
        return metadata

    """
    Using just the part ID and session ID get if the participant is PART.1 or PART.2
    """
    @staticmethod
    def get_name_for_part(part_id, session_id, split):
        sessions_metadata = UDIVA.read_metadata("sessions", split) # load metadata
        jdx = np.where(sessions_metadata[:,0] == session_id)[0][0]
        if sessions_metadata[jdx,1] == part_id:
            name = "PART.1"
        elif sessions_metadata[jdx,2] == part_id:
            name = "PART.2"
        return name

    """
    Gets the time intervals from the utterances of a given participant ID
    """
    @staticmethod
    def get_times_for_part(part_id, session_id, task, split):
        logging.info(f"Reading time intervals for participant {part_id} from session {session_id}, task {task}, split {split}")
        # build the path for reading the transcript
        path = os.path.join(DATA_DIR, split, "transcriptions")
        extension = f"_{task}.srt"
        session_id_str = str(int(session_id)).rjust(6,'0')
        file_name = os.path.join(path, session_id_str, f"{session_id_str}{extension}")
        logging.info(f"Reading transcript file from: {file_name}")
        # read session transcript
        with codecs.open(file_name, 'r', encoding="utf-8", errors="ignore") as reader:
            stream = reader.read()
        # check if the participant is PART.1 or PART.2 in the session
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        # save the start and end times
        start = []
        end = []
        partname = []
        for sub in srt.parse(stream):
            if sub.content[:len('PART.X')] == part_name:
                partname.append(sub.content[:len('PART.X')])
                start.append(sub.start.total_seconds())
                end.append(sub.end.total_seconds())
        times = np.column_stack((partname,start,end))
        return times

    """
    Gets the time intervals from the utterances of both participants
    This will be used when extracting features for a participant even in the interactions of the other participant
    """
    @staticmethod
    def get_times_for_both_parts(session_id, task, split):
        logging.info(f"Reading time intervals for participant from session {session_id}, task {task}, split {split}")
        # build the path for reading the transcript
        path = os.path.join(DATA_DIR, split, "transcriptions")
        extension = f"_{task}.srt"
        session_id_str = str(int(session_id)).rjust(6,'0')
        file_name = os.path.join(path, session_id_str, f"{session_id_str}{extension}")
        logging.info(f"Reading transcript file from: {file_name}")
        # read session transcript
        with codecs.open(file_name, 'r', encoding="utf-8", errors="ignore") as reader:
            stream = reader.read()
        # save the start and end times
        start = []
        end = []
        partname = []
        for sub in srt.parse(stream):
            if sub.content[:len("PART")] == "PART":
                partname.append(sub.content[:len('PART.X')])
                start.append(sub.start.total_seconds())
                end.append(sub.end.total_seconds())
        times = np.column_stack((partname,start,end))
        return times

    """
    Methods to extract features from video
    """

    """
    Builds the path for the HDF5 file
    """
    @staticmethod
    def generate_hdf5_filename(part_name, session_id, task, split):
        logging.info(f"Generating HDF5 filename for {part_name} from session {session_id}, task {task}, split {split}")
        part_num = PART_NUMBERS[part_name] 
        name = str(session_id).rjust(6,'0')
        path = os.path.join(DATA_DIR, split, "annotations")
        if task == 'talk' and split == 'val' or task == 'talk' and split == 'test':
            file_name = os.path.join(path,name,f"FC{part_num}_{TASK_LETTERS[task]}",f"annotations_cleaned.hdf5")
        else:
            file_name = os.path.join(path,name,f"FC{part_num}_{TASK_LETTERS[task]}",f"annotations_raw.hdf5")
        logging.info(f"Generated filename: {file_name}")
        return file_name

    @staticmethod
    def build_repr_landmarks_part_face(part_id, session_id, task, split):
        logging.info(f"Building landmarks representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        times = UDIVA.get_times_for_part(part_id, session_id, task, split) # needed for the repr slices
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
        
        with h5py.File(file_name, 'r') as reader:
            sequence = [] # stores all the sequence of utterances
            frames = list(reader.keys())
            logging.info(f"Total frames in the session: {len(frames)}")
            logging.info(f'Sampling at {FPS} FPS')          
            for utterance in times:
                # computing times -> seconds to frames
                start_time = float(utterance[1])
                end_time = float(utterance[2])
                logging.info(f"Start time: {start_time}, end time: {end_time}")
                total_time = end_time - start_time
                logging.info(f"Total time {total_time}")
                start_frame = int(FPS * start_time)
                end_frame = int(FPS * end_time)
                total_frames = int(FPS * total_time)
                logging.info(f"Frames in that period of time {total_frames}")
                logging.info(f"Start frame {start_frame}, end frame {end_frame}")
                # vectors to store the sum for computing mean in the future
                face_frames = np.empty((0, 204))
                # start storing relevant frames
                for i in range(start_frame, end_frame+1):
                    try: # since the compute of the frames could take some index out of range
                        data = reader[frames[i]] 
                    except IndexError:
                        break
                    if not data.keys():
                        continue
                    if data.attrs['valid'] == False:
                        logging.info('False frame')
                    face = data['face']
                    if 'landmarks' in face.keys() and face.attrs['valid'] == True:
                        flattened = np.ndarray.flatten(np.array(list(face['landmarks'])))
                        face_frames = np.vstack((face_frames, flattened))
                # compute the mean for every vector
                face_mean = np.mean(face_frames, axis=0)
                # compute the std for every vector
                face_std = np.std(face_frames, axis=0)
                # concatenate both the mean and the std vectors
                sample = np.concatenate((face_mean, face_std))
                # store in the sequence
                sequence.append(sample)
        sequence = np.asarray(sequence)
        return sequence

    """
    Builds the representation per utterance for participant
    """
    @staticmethod
    def build_repr_landmarks_part(part_id, session_id, task, split):
        logging.info(f"Building landmarks representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        times = UDIVA.get_times_for_part(part_id, session_id, task, split) # needed for the repr slices
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
        
        with h5py.File(file_name, 'r') as reader:
            sequence = [] # stores all the sequence of utterances
            frames = list(reader.keys())
            logging.info(f"Total frames in the session: {len(frames)}")
            logging.info(f'Sampling at {FPS} FPS')          
            for utterance in times:
                # computing times -> seconds to frames
                start_time = float(utterance[1])
                end_time = float(utterance[2])
                logging.info(f"Start time: {start_time}, end time: {end_time}")
                total_time = end_time - start_time
                logging.info(f"Total time {total_time}")
                start_frame = int(FPS * start_time)
                end_frame = int(FPS * end_time)
                total_frames = int(FPS * total_time)
                logging.info(f"Frames in that period of time {total_frames}")
                logging.info(f"Start frame {start_frame}, end frame {end_frame}")
                # vectors to represent each mean
                body_frames = np.empty((0, 72))
                face_frames = np.empty((0, 204))
                gaze_frames = np.empty((0, 3))
                righthand_frames = np.empty((0, 63))
                lefthand_frames = np.empty((0, 63))
                # start storing relevant frames
                for i in range(start_frame, end_frame+1):
                    try: # since the compute of the frames could take some index out of range
                        data = reader[frames[i]] 
                    except IndexError:
                        break
                    if not data.keys():
                        continue
                    if data.attrs['valid'] == False:
                        logging.info('False frame')
                    body = data['body']
                    if 'landmarks' in body.keys() and body.attrs['valid'] == True:
                        flattened = np.ndarray.flatten(np.array(list(body['landmarks'])))
                        body_frames = np.vstack((body_frames, flattened))
                    face = data['face']
                    if 'landmarks' in face.keys() and face.attrs['valid'] == True:
                        flattened = np.ndarray.flatten(np.array(list(face['landmarks'])))
                        face_frames = np.vstack((face_frames, flattened))
                        gaze_frames = np.vstack((gaze_frames, list(face.attrs['gaze'])))
                    hands = data['hands']
                    righthand = hands['right']
                    if 'landmarks' in righthand.keys() and righthand.attrs['valid'] == True and righthand.attrs['visible'] == True:
                        flattened = np.ndarray.flatten(np.array(list(righthand['landmarks'])))
                        righthand_frames = np.vstack((righthand_frames, flattened))
                    lefthand = hands['left']
                    if 'landmarks' in lefthand.keys() and lefthand.attrs['valid'] == True and lefthand.attrs['visible'] == True:
                        flattened = np.ndarray.flatten(np.array(list(lefthand['landmarks'])))
                        lefthand_frames = np.vstack((lefthand_frames, flattened))           
                # compute the mean for every vector
                body_mean = np.nanmean(body_frames, axis=0)
                face_mean = np.nanmean(face_frames, axis=0)
                gaze_mean = np.nanmean(gaze_frames, axis=0)
                righthand_mean = np.nanmean(righthand_frames, axis=0)
                lefthand_mean = np.nanmean(lefthand_frames, axis=0)
                # concatenate the vectors 
                sample_mean = np.concatenate((body_mean, face_mean, gaze_mean, righthand_mean, lefthand_mean))
                # compute the std for every vector
                body_std = np.nanstd(body_frames, axis=0)
                face_std = np.nanstd(face_frames, axis=0)
                gaze_std = np.nanstd(gaze_frames, axis=0)
                righthand_std = np.nanstd(righthand_frames, axis=0)
                lefthand_std = np.nanstd(lefthand_frames, axis=0)
                # concatenate them
                sample_std = np.concatenate((body_std, face_std, gaze_std, righthand_std, lefthand_std))
                # concatenate both the mean and the std vectors
                sample = np.concatenate((sample_mean, sample_std))
                # store in the sequence
                sequence.append(sample)
        sequence = np.asarray(sequence)
        return sequence

    """
    Builds the representation for one participant considering the utterances from both participants
    """
    @staticmethod
    def build_repr_landmarks_part_both(part_id, session_id, task, split):
        logging.info(f"Building landmarks representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        times = UDIVA.get_times_for_both_parts(session_id, task, split) # needed for the repr slices
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        file_name = UDIVA.generate_hdf5_filename(part_name, session_id, task, split)
        
        with h5py.File(file_name, 'r') as reader:
            sequence = [] # stores all the sequence of utterances
            frames = list(reader.keys())
            logging.info(f"Total frames in the session: {len(frames)}")
            logging.info(f'Sampling at {FPS} FPS')          
            for utterance in times:
                # computing times -> seconds to frames
                start_time = float(utterance[1])
                end_time = float(utterance[2])
                logging.info(f"Start time: {start_time}, end time: {end_time}")
                total_time = end_time - start_time
                logging.info(f"Total time {total_time}")
                start_frame = int(FPS * start_time)
                end_frame = int(FPS * end_time)
                total_frames = int(FPS * total_time)
                logging.info(f"Frames in that period of time {total_frames}")
                logging.info(f"Start frame {start_frame}, end frame {end_frame}")
                # vectors to represent each mean
                body_frames = np.empty((0, 72))
                face_frames = np.empty((0, 204))
                righthand_frames = np.empty((0, 63))
                lefthand_frames = np.empty((0, 63))
                gaze_frames = np.empty((0, 3))
                # start storing relevant frames
                for i in range(start_frame, end_frame+1):
                    try: # since the compute of the frames could take some index out of range
                        data = reader[frames[i]] 
                    except IndexError:
                        break
                    if not data.keys():
                        continue
                    if data.attrs['valid'] == False:
                        logging.info('False frame')
                    body = data['body']
                    if 'landmarks' in body.keys() and body.attrs['valid'] == True:
                        flattened = np.ndarray.flatten(np.array(list(body['landmarks'])))
                        body_frames = np.vstack((body_frames, flattened))
                    face = data['face']
                    if 'landmarks' in face.keys() and face.attrs['valid'] == True:
                        flattened = np.ndarray.flatten(np.array(list(face['landmarks'])))
                        face_frames = np.vstack((face_frames, flattened))
                        gaze_frames = np.vstack((gaze_frames, list(face.attrs['gaze'])))
                    hands = data['hands']
                    righthand = hands['right']
                    if 'landmarks' in righthand.keys() and righthand.attrs['valid'] == True and righthand.attrs['visible'] == True:
                        flattened = np.ndarray.flatten(np.array(list(righthand['landmarks'])))
                        righthand_frames = np.vstack((righthand_frames, flattened))
                    lefthand = hands['left']
                    if 'landmarks' in lefthand.keys() and lefthand.attrs['valid'] == True and lefthand.attrs['visible'] == True:
                        flattened = np.ndarray.flatten(np.array(list(lefthand['landmarks'])))
                        lefthand_frames = np.vstack((lefthand_frames, flattened))           
                # compute the mean for every vector
                body_mean = np.mean(body_frames, axis=0)
                face_mean = np.mean(face_frames, axis=0)
                gaze_mean = np.mean(gaze_frames, axis=0)
                righthand_mean = np.mean(righthand_frames, axis=0)
                lefthand_mean = np.mean(lefthand_frames, axis=0)
                # concatenate the vectors 
                sample_mean = np.concatenate((body_mean, face_mean, gaze_mean, righthand_mean, lefthand_mean))
                # compute the std for every vector
                body_std = np.std(body_frames, axis=0)
                face_std = np.std(face_frames, axis=0)
                gaze_std = np.std(gaze_frames, axis=0)
                righthand_std = np.std(righthand_frames, axis=0)
                lefthand_std = np.std(lefthand_frames, axis=0)
                # concatenate them
                sample_std = np.concatenate((body_std, face_std, gaze_std, righthand_std, lefthand_std))
                # concatenate both the mean and the std vectors
                sample = np.concatenate((sample_mean, sample_std))
                # store in the sequence
                sequence.append(sample)
        sequence = np.asarray(sequence)
        return sequence

    """
    Builds all the representation for a full set split
    """
    @staticmethod
    def build_repr_landmarks_split(split):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        repr.append(UDIVA.build_repr_landmarks_part(part, int(session), task, split))
            parts_repr.append(repr)
        parts_repr = np.asarray(parts_repr, dtype=object)
        return parts_repr

    """
    Builds all the representation for a full set split
    """
    @staticmethod
    def build_repr_landmarks_split_face(split):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        repr.append(UDIVA.build_repr_landmarks_part_face(part, int(session), task, split))
            parts_repr.append(repr)
        parts_repr = np.asarray(parts_repr, dtype=object)
        return parts_repr

    """
    Methods to extract features from audio
    """

    """
    Builds the path for the WAV audio file 
    """
    @staticmethod
    def generate_wav_filename(part_name, session_id, task, split):
        logging.info(f"Generating WAV filename for {part_name} from session {session_id}, task {task}, split {split}")
        part_num = PART_NUMBERS[part_name] 
        name = str(session_id).rjust(6,'0')
        path = os.path.join(DATA_DIR, split, "recordings")
        file_name = os.path.join(path,name,f"FC{part_num}_{TASK_LETTERS[task]}.wav")
        logging.info(f"Generated filename: {file_name}")
        return file_name

    @staticmethod
    def CreateVGGishNetwork(sess, hop_size=0.96):   # Hop size is in seconds.
        """Define VGGish model, load the checkpoint, and return a dictionary that points
        to the different tensors defined by the model.
        """
        vggish_slim.define_vggish_slim()
        # checkpoint_path = 'vggish_model.ckpt'
        checkpoint_path = os.path.join(ROOT_DIR, "..", "models", "vggish", "vggish_model.ckpt")
        vggish_params.EXAMPLE_HOP_SECONDS = hop_size
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        layers = {'conv1': 'vggish/conv1/Relu',
                    'pool1': 'vggish/pool1/MaxPool',
                    'conv2': 'vggish/conv2/Relu',
                    'pool2': 'vggish/pool2/MaxPool',
                    'conv3': 'vggish/conv3/conv3_2/Relu',
                    'pool3': 'vggish/pool3/MaxPool',
                    'conv4': 'vggish/conv4/conv4_2/Relu',
                    'pool4': 'vggish/pool4/MaxPool',
                    'fc1': 'vggish/fc1/fc1_2/Relu',
                    #'fc2': 'vggish/fc2/Relu',
                    'embedding': 'vggish/embedding',
                    'features': 'vggish/input_features',
                }
        g = tf.get_default_graph()
        for k in layers:
            layers[k] = g.get_tensor_by_name( layers[k] + ':0')
        return {'features': features_tensor,
                'embedding': embedding_tensor,
                'layers': layers,
                }

    @staticmethod
    def ProcessWithVGGish(vgg, x, sr, sess):
        '''Run the VGGish model, starting with a sound (x) at sample rate
        (sr). Return a whitened version of the embeddings. Sound must be scaled to be
        floats between -1 and +1.'''
        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.waveform_to_examples(x, sr)
        # print('Log Mel Spectrogram example: ', input_batch[0])
        [embedding_batch] = sess.run([vgg['embedding']],
                                    feed_dict={vgg['features']: input_batch})
        # Postprocess the results to produce whitened quantized embeddings.
        # pca_params_path = 'vggish_pca_params.npz'
        pca_params_path = os.path.join(ROOT_DIR, "..", "models", "vggish", "vggish_pca_params.npz")
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        # print('Postprocessed VGGish embedding: ', postprocessed_batch[0])
        return postprocessed_batch[0]

    @staticmethod
    def load_audio_model():
        tf.compat.v1.disable_eager_execution()
        tf.reset_default_graph()
        sess = tf.Session()
        vgg = UDIVA.CreateVGGishNetwork(sess=sess)
        return vgg, sess

    """
    Builds the audio representation for one participant using the VGGish model
    """
    @staticmethod
    def build_repr_audio_part(part_id, session_id, task, split, vgg, sess):
        logging.info(f"Building audio representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        times = UDIVA.get_times_for_part(part_id, session_id, task, split) # needed for the repr slices
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        # stores the embeddings recovered from the audio
        embeddings_list = []
        # get video name
        file_name = UDIVA.generate_wav_filename(part_name, session_id, task, split)
        print("File name:", file_name)
        # load audio clip
        data, sr = librosa.load(file_name) # sr is the sampling rate
        # create empty vector of the size of the VGGish embeddings
        embeddings = np.empty((0,128), dtype=np.uint8)
        zeros = np.zeros([128], dtype=np.uint8)
        # compute the embedding for each utterance
        for utterance in times:
            # computing times -> seconds to samples
            start_time = float(utterance[1])
            end_time = float(utterance[2])
            logging.info(f"Start time: {start_time}, end time: {end_time}")
            total_time = end_time - start_time
            logging.info(f"Total time {total_time}")
            start_index = int(float(start_time*sr))
            end_index = int(float(end_time)*sr)
            total_samples = end_index - start_index
            logging.info(f"Samples in that period of time {total_samples}")
            logging.info(f"Start frame {start_index}, end frame {end_index}")
            # if there's an audio shorter than a second (VGGIsh problem)
            if end_index - start_index <= sr: 
                embeddings = np.vstack((embeddings, zeros))
                continue
            # take the slice
            audio = data[start_index:end_index]
            # compute the embedding using the VGGish model
            # vgg, sess = UDIVA.load_audio_model()
            try:
                embedding = UDIVA.ProcessWithVGGish(vgg, audio, sr, sess)
                embeddings = np.vstack((embeddings, embedding))
            except IndexError: # for index errors where the processwithvggish returns nothing
                embeddings = np.vstack((embeddings, zeros))
            # store the utterance representation
        # embeddings_list = np.asarray(embeddings_list)
        return embeddings

    @staticmethod
    def build_repr_audio_split(split, vgg, sess):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        repr.append(UDIVA.build_repr_audio_part(part, int(session), task, split, vgg, sess))
            parts_repr.append(repr)
        parts_repr = np.asarray(parts_repr, dtype=object)
        return parts_repr

    """
    Methods to extract features from text
    """

    @staticmethod
    def tokenize_utterance(utterance):
        utterance = utterance.lower()
        utterance = re.sub('([^\x00-\xFF])+', '', utterance) # remove non UTF-8 characters https://www.utf8-chartable.de/
        utterance = re.sub('[^\w\s]', '', utterance) # remove punctuation https://datagy.io/python-remove-punctuation-from-string/#:~:text=One%20of%20the%20easiest%20ways,maketrans()%20method.
        utterance = utterance.split()
        utterance = [word for word in utterance if word != ''] # removes empty spaces
        return utterance 

    @staticmethod
    def vectorize_utterance(utterance, text_model):
        vector = np.zeros([300])
        if not utterance: # for empty utterances after tokenization
            return vector
        for word in utterance:
            vector = np.add(vector, text_model.wv[word])
        vector = vector * (1 / len(utterance)) # return the mean from word vectors
        return vector

    @staticmethod
    def translate_transcript(transcript, lang):
        print(f"Translating session from {lang} to Spanish")
        langs = {"English":"en", "Catalan":"ca"}
        translator = Translator()
        subs = []
        for utterance in transcript:
            translation = translator.translate(utterance, src=langs[lang], dest='es')
            subs.append(translation.text)
        return subs

    @staticmethod
    def read_transcript(session_id, task, split):    
        print(f'Reading transcriptions from "{task}" task from {split} split')
        path = os.path.join(DATA_DIR, split, "transcriptions")
        extension = f'_{task}.srt'
        str_session_id = str(session_id).rjust(6,'0')
        file_name = os.path.join(path, str_session_id, f"{str_session_id}{extension}")
        print(f'Processing {file_name}')
        with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as file:
            stream = file.read()
        return stream

    @staticmethod
    def parse_transcript(session_transcript):    
        subs = []
        for sub in srt.parse(session_transcript):
            subs.append(sub.content)
        return subs

    @staticmethod
    def separate_speaker(parsed_transcript, part_name):
        subs = []
        for utterance in parsed_transcript:
            if utterance[0:len("PART.X")] == part_name:
                subs.append(utterance[len(part_name)+2:]) # cuts the PART.X part from the sub
        return subs

    @staticmethod 
    def load_text_model():
        print('Loading Fasttext model')
        path = os.path.join(ROOT_DIR,"..","models","fasttext","fasttext-sbwc.bin")
        text_model = fasttext.load_facebook_model(path)
        print('Finished loading Fasttext model')
        return text_model

    """ 
    Builds the representation for the text modality for a given participant 
    """
    @staticmethod 
    def build_repr_text_part(part_id, session_id, task, split, text_model):
        print(f"Building textual representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        # read transcript and just store utterances from the part_id
        stream = UDIVA.read_transcript(session_id, task, split)
        transcript = UDIVA.parse_transcript(stream)
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        transcript = UDIVA.separate_speaker(transcript, part_name)
        # check the language of the session
        sessions_metadata = UDIVA.read_metadata("sessions", split)
        i = np.where(sessions_metadata[:,0] == session_id)[0][0]
        lang = sessions_metadata[i][3] # read the language column of the session ID
        print("Language of the session is:", lang)
        # translate session if it is not in Spanish
        if lang != "Spanish":
            transcript = UDIVA.translate_transcript(transcript, lang)
        # vectorize the transcript
        repr = []
        for utterance in transcript:
            tokenized_utterance = UDIVA.tokenize_utterance(utterance)
            repr.append(UDIVA.vectorize_utterance(tokenized_utterance, text_model))
        repr = np.asarray(repr)
        return repr
    
    @staticmethod
    def build_repr_text_split(split, text_model):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        repr.append(UDIVA.build_repr_text_part(part, int(session), task, split, text_model))
            parts_repr.append(repr)
        parts_repr = np.asarray(parts_repr, dtype=object)
        return parts_repr


    """
    Builds representations for BETO
    """

    """
    Returns a list of lists. Each list is made of the words in a utterance.
    This returns the tokenized utterance.
    """
    @staticmethod 
    def build_repr_text_part_BETO(part_id, session_id, task, split):
        print(f"Building textual representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        # read transcript and just store utterances from the part_id
        stream = UDIVA.read_transcript(session_id, task, split)
        transcript = UDIVA.parse_transcript(stream)
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        transcript = UDIVA.separate_speaker(transcript, part_name)
        # check the language of the session
        sessions_metadata = UDIVA.read_metadata("sessions", split)
        i = np.where(sessions_metadata[:,0] == session_id)[0][0]
        lang = sessions_metadata[i][3] # read the language column of the session ID
        print("Language of the session is:", lang)
        # translate session if it is not in Spanish
        if lang != "Spanish":
            transcript = UDIVA.translate_transcript(transcript, lang)
        # vectorize the transcript
        repr = []
        for utterance in transcript:
            tokenized_utterance = UDIVA.tokenize_utterance(utterance)
            repr.append(tokenized_utterance)
        return repr


    """
    Returns a list of lists. Each list is made of an utterance.
    """
    @staticmethod 
    def build_repr_text_part_BETO_fine(part_id, session_id, task, split):
        print(f"Building textual representation for participant {part_id} from session {session_id}, task {task}, split {split}")
        # read transcript and just store utterances from the part_id
        stream = UDIVA.read_transcript(session_id, task, split)
        transcript = UDIVA.parse_transcript(stream)
        part_name = UDIVA.get_name_for_part(part_id, session_id, split)
        transcript = UDIVA.separate_speaker(transcript, part_name)
        # check the language of the session
        sessions_metadata = UDIVA.read_metadata("sessions", split)
        i = np.where(sessions_metadata[:,0] == session_id)[0][0]
        lang = sessions_metadata[i][3] # read the language column of the session ID
        print("Language of the session is:", lang)
        # translate session if it is not in Spanish
        if lang != "Spanish":
            transcript = UDIVA.translate_transcript(transcript, lang)
        return transcript

    """
    Builds the entire representation for a split containing tokenized utterances
    """
    @staticmethod
    def build_repr_text_split_BETO(split):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        rep_part = UDIVA.build_repr_text_part_BETO(part, int(session), task, split)
                        for rep in rep_part:
                            repr.append(rep)
            for rep in repr:
                parts_repr.append(rep)
        return parts_repr

    """
    Builds the entire representation for a split containing just the utterances
    """
    @staticmethod
    def build_repr_text_split_BETO_fine(split):
        parts_metadata = UDIVA.read_metadata('parts', split)
        parts_repr = []
        labels = []
        for i, part in enumerate(parts_metadata[:,0]):
            repr = []
            for session in parts_metadata[i,1:6]:
                if math.isnan(session):
                    continue
                else:
                    for task in TASKS:
                        rep_part = UDIVA.build_repr_text_part_BETO_fine(part, int(session), task, split)
                        for rep in rep_part:
                            repr.append(rep)
            for rep in repr:
                parts_repr.append(rep)
                labels.append(parts_metadata[i,6::])
        return parts_repr, labels

    """
    Methods to generate samples for a RNN
    """

    """
    Generates the timesteps for a given participant 
    """
    @staticmethod
    def build_timesteps(task_vectors, metadata, part_i, n_timesteps=10, overlapping=3):
        repr = []
        labels = []
        for i in range(0, len(task_vectors), overlapping):
            if (i+(n_timesteps-1)) < len(task_vectors):
                timestep = []
                for j in range(n_timesteps):
                    timestep.append(task_vectors[i+j])
                repr.append(timestep)
                labels.append(metadata[part_i][6::])
            else:
                break
        return repr, labels

    """
    Generates the timesteps for a given split 
    """
    @staticmethod
    def build_samples(features, metadata, n_timesteps, overlapping):
        samples = []
        labels = []
        for i, part in enumerate(features):
            for task in part:
                repr_samples, repr_labels = UDIVA.build_timesteps(task, metadata, i, n_timesteps, overlapping)
                for j in range(len(repr_samples)):
                    samples.append(repr_samples[j])
                    labels.append(repr_labels[j])
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        return samples, labels