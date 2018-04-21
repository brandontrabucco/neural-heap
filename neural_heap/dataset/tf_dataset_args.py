import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset import TRAIN_EXAMPLES
from neural_heap.dataset import VAL_EXAMPLES
from neural_heap.dataset import DATASET_COLUMNS
from neural_heap.dataset import DATASET_RANGE
from neural_heap.dataset import DATASET_FILENAMES
from neural_heap.dataset import MIN_RATIO
from neural_heap.dataset import DATASET_DEFAULT
from neural_heap.dataset import BATCH_SIZE
from neural_heap.dataset import TRAIN_EPOCH_SIZE
from neural_heap.dataset import VAL_EPOCH_SIZE
from neural_heap.dataset import NUM_THREADS
import argparse

class TFDatasetArgs(object):

    def __init__(
            self,
            parser=None,
            name="TFDatasetArgs"):
        self.parser = parser
        self.name = name
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description=self.name)
        self.parser.add_argument(
            "--train_dataset",
            type=str,
            default=DATASET_FILENAMES["train"])
        self.parser.add_argument(
            "--val_dataset",
            type=str,
            default=DATASET_FILENAMES["val"])
        self.parser.add_argument(
            "--train_instances",
            type=int,
            default=TRAIN_EXAMPLES)
        self.parser.add_argument(
            "--val_instances",
            type=int,
            default=VAL_EXAMPLES)
        self.parser.add_argument(
            "--min_ratio",
            type=float,
            default=MIN_RATIO)
        self.parser.add_argument(
            "--dataset_columns",
            type=int,
            default=DATASET_COLUMNS)
        self.parser.add_argument(
            "--dataset_range",
            type=int,
            default=DATASET_RANGE)
        self.parser.add_argument(
            "--dataset_default",
            type=int,
            default=DATASET_DEFAULT)
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE)
        self.parser.add_argument(
            "--train_epoch_size",
            type=int,
            default=TRAIN_EPOCH_SIZE)
        self.parser.add_argument(
            "--val_epoch_size",
            type=int,
            default=VAL_EPOCH_SIZE)
        self.parser.add_argument(
            "--num_threads",
            type=int,
            default=NUM_THREADS)

    def get_parser(
            self):
        return self.parser
