import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")

from neural_heap.dataset import DATASET_FILENAMES
from neural_heap.dataset.io_synthesis import TRAIN_EXAMPLES
from neural_heap.dataset.io_synthesis import VAL_EXAMPLES
from neural_heap.dataset import MIN_RATIO
from neural_heap.dataset.io_synthesis import DATASET_COLUMNS
from neural_heap.dataset.io_synthesis import DATASET_RANGE
from neural_heap.dataset import DATASET_DEFAULT
from neural_heap.heap import HEAP_NAME
from neural_heap.heap import HEAP_SIZE
from neural_heap.heap import OP_SIZE

from neural_heap.dataset import BATCH_SIZE
from neural_heap.dataset import TRAIN_EPOCH_SIZE
from neural_heap.dataset import VAL_EPOCH_SIZE
from neural_heap.dataset import NUM_THREADS
from neural_heap.graph import TOTAL_LOGS
from neural_heap import CHECKPOINT_BASEDIR
from neural_heap import PLOTS_BASEDIR

from neural_heap import PREFIX_TOTAL
from neural_heap import PREFIX_CONTROLLER
from neural_heap import PREFIX_OUTPUT
from neural_heap import PREFIX_OPERATOR
from neural_heap import PREFIX_OPERAND
from neural_heap import EXTENSION_NUMBER
from neural_heap import EXTENSION_LOSS
from neural_heap import EXTENSION_WEIGHTS
from neural_heap import EXTENSION_BIASES
from neural_heap import COLLECTION_LOSSES
from neural_heap import COLLECTION_PARAMETERS
from neural_heap import COLLECTION_ACTIVATIONS

from neural_heap import LOSS_OP
from neural_heap import GLOBAL_STEP_OP
from neural_heap import PREDICTION_OP
from neural_heap import INPUTS_OP
from neural_heap import LABELS_OP
from neural_heap import PREDICTION_DETOKENIZED_OP
from neural_heap import INPUTS_DETOKENIZED_OP
from neural_heap import LABELS_DETOKENIZED_OP
from neural_heap import TRAIN_OP
from neural_heap import VAL_OP

from neural_heap.graph import ENSEMBLE_SIZE
from neural_heap.graph import LSTM_DEPTH
from neural_heap.graph import LSTM_SIZE
from neural_heap.graph import DROPOUT_PROBABILITY
from neural_heap.graph import USE_DROPOUT
from neural_heap.graph import INITIAL_LEARNING_RATE
from neural_heap.graph import DECAY_STEPS
from neural_heap.graph import DECAY_FACTOR

class TFConfig(object):

    def __init__(self):
        self.DATASET_FILENAMES = DATASET_FILENAMES
        self.TRAIN_EXAMPLES = TRAIN_EXAMPLES
        self.VAL_EXAMPLES = VAL_EXAMPLES
        self.MIN_RATIO = MIN_RATIO
        self.DATASET_COLUMNS = DATASET_COLUMNS
        self.DATASET_RANGE = DATASET_RANGE
        self.DATASET_DEFAULT = DATASET_DEFAULT
        self.HEAP_NAME = HEAP_NAME
        self.HEAP_SIZE = HEAP_SIZE
        self.OP_SIZE = OP_SIZE

        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_EPOCH_SIZE = TRAIN_EPOCH_SIZE
        self.VAL_EPOCH_SIZE = VAL_EPOCH_SIZE
        self.NUM_THREADS = NUM_THREADS
        self.TOTAL_LOGS = TOTAL_LOGS
        self.CHECKPOINT_BASEDIR = CHECKPOINT_BASEDIR
        self.PLOTS_BASEDIR = PLOTS_BASEDIR

        self.PREFIX_TOTAL = PREFIX_TOTAL
        self.PREFIX_CONTROLLER = PREFIX_CONTROLLER
        self.PREFIX_OUTPUT = PREFIX_OUTPUT
        self.PREFIX_OPERATOR = PREFIX_OPERATOR
        self.PREFIX_OPERAND = PREFIX_OPERAND
        self.EXTENSION_NUMBER = EXTENSION_NUMBER
        self.EXTENSION_LOSS = EXTENSION_LOSS
        self.EXTENSION_WEIGHTS = EXTENSION_WEIGHTS
        self.EXTENSION_BIASES = EXTENSION_BIASES
        self.COLLECTION_LOSSES = COLLECTION_LOSSES
        self.COLLECTION_PARAMETERS = COLLECTION_PARAMETERS
        self.COLLECTION_ACTIVATIONS = COLLECTION_ACTIVATIONS

        self.LOSS_OP = LOSS_OP
        self.GLOBAL_STEP_OP = GLOBAL_STEP_OP
        self.PREDICTION_OP = PREDICTION_OP
        self.INPUTS_OP = INPUTS_OP
        self.LABELS_OP = LABELS_OP
        self.PREDICTION_DETOKENIZED_OP = PREDICTION_DETOKENIZED_OP
        self.INPUTS_DETOKENIZED_OP = INPUTS_DETOKENIZED_OP
        self.LABELS_DETOKENIZED_OP = LABELS_DETOKENIZED_OP
        self.TRAIN_OP = TRAIN_OP
        self.VAL_OP = VAL_OP

        self.ENSEMBLE_SIZE = ENSEMBLE_SIZE
        self.LSTM_SIZE = LSTM_SIZE
        self.LSTM_DEPTH = LSTM_DEPTH
        self.DROPOUT_PROBABILITY = DROPOUT_PROBABILITY
        self.USE_DROPOUT = USE_DROPOUT
        self.INITIAL_LEARNING_RATE = INITIAL_LEARNING_RATE
        self.DECAY_STEPS = DECAY_STEPS
        self.DECAY_FACTOR = DECAY_FACTOR

    def __call__(self, num_epoch, offset):
        self.NUM_EPOCH = num_epoch
        self.TRAIN_ITERATIONS = (self.TRAIN_EPOCH_SIZE * self.NUM_EPOCH)
        self.TRAIN_STOP_AT_STEP = self.TRAIN_ITERATIONS + offset
        self.VAL_ITERATIONS = self.VAL_EPOCH_SIZE
        self.VAL_STOP_AT_STEP = self.VAL_ITERATIONS + offset