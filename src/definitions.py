import os

ROOT_DIR = os.path.realpath(os.path.dirname(__file__)) # ~/udiva/code/src/
DATA_DIR = os.path.join(ROOT_DIR,'..','..','dataset') # ~/udiva/dataset/
CODE_DIR = os.path.join(ROOT_DIR,'..') # ~/udiva/code/

TASKS = ["talk", "lego", "animals", "ghost"]
SPLITS = ["train", "val", "test"]
SOURCES = ["recordings", "annotations", "transcriptions"]
PART_NUMBERS = {"PART.1":1, "PART.2":2}
TASK_LETTERS = {'talk':'T', 'lego':'L', 'animals':'A', 'ghost':'G'}
FPS = 25