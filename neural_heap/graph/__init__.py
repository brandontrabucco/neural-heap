import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset.io_synthesis import DATASET_RANGE
from neural_heap.dataset.io_synthesis import TRAIN_EXAMPLES
from neural_heap.dataset import BATCH_SIZE
TOTAL_LOGS = 10
ENSEMBLE_SIZE = 3
LSTM_SIZE = (DATASET_RANGE * 4 * ENSEMBLE_SIZE)
LSTM_DEPTH = 4
DROPOUT_PROBABILITY = (1 / ENSEMBLE_SIZE)
USE_DROPOUT = True
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
DECAY_FACTOR = 0.5