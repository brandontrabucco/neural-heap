import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset.io_synthesis import TRAIN_EXAMPLES
from neural_heap.dataset.io_synthesis import VAL_EXAMPLES
from neural_heap.dataset.io_synthesis import DATASET_COLUMNS
from neural_heap.dataset.io_synthesis import DATASET_RANGE
import argparse

class IOSynthesisArgs(object):

    def __init__(
            self,
            parser=None,
            name="IOSynthesisArgs"):
        self.parser = parser
        self.name = name
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description=self.name)
        self.parser.add_argument(
            "--range",
            type=int,
            default=DATASET_RANGE)
        self.parser.add_argument(
            "--length",
            type=int,
            default=DATASET_COLUMNS)
        self.parser.add_argument(
            "--train_instances",
            type=int,
            default=TRAIN_EXAMPLES)
        self.parser.add_argument(
            "--val_instances",
            type=int,
            default=VAL_EXAMPLES)

    def get_parser(
            self):
        return self.parser
