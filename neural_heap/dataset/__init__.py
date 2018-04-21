import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset.io_synthesis import TRAIN_EXAMPLES
from neural_heap.dataset.io_synthesis import VAL_EXAMPLES
from neural_heap.dataset.io_synthesis import DATASET_COLUMNS
from neural_heap.dataset.io_synthesis import DATASET_RANGE

DATASET_FILENAMES = {
    "train": ["neural_heap/dataset/csv/train_dataset.csv"],
    "val": ["neural_heap/dataset/csv/val_dataset.csv"]}
MIN_RATIO = 10
DATASET_DEFAULT = DATASET_RANGE // 2
BATCH_SIZE = 128
TRAIN_EPOCH_SIZE = TRAIN_EXAMPLES // BATCH_SIZE
VAL_EPOCH_SIZE = VAL_EXAMPLES // BATCH_SIZE
NUM_THREADS = 2