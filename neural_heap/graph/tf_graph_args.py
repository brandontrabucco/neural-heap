import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.graph import ENSEMBLE_SIZE
from neural_heap.graph import LSTM_SIZE
from neural_heap.graph import LSTM_DEPTH
from neural_heap.graph import DROPOUT_PROBABILITY
from neural_heap.graph import USE_DROPOUT
from neural_heap.graph import INITIAL_LEARNING_RATE
from neural_heap.graph import DECAY_STEPS
from neural_heap.graph import DECAY_FACTOR
import argparse

class TFGraphArgs(object):

    def __init__(
            self,
            parser=None,
            name="TFGraphArgs",
            range_default=32,
            train_instances=10000,
            batch_size=128):
        self.parser = parser
        self.name = name
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description=self.name)
        self.parser.add_argument(
            "--ensemble_size",
            type=int,
            default=ENSEMBLE_SIZE)
        self.parser.add_argument(
            "--lstm_size",
            type=int,
            default=LSTM_SIZE)
        self.parser.add_argument(
            "--lstm_depth",
            type=int,
            default=LSTM_DEPTH)
        self.parser.add_argument(
            "--dropout_probability",
            type=float,
            default=DROPOUT_PROBABILITY)
        self.parser.add_argument(
            "--use_dropout",
            type=bool,
            default=USE_DROPOUT)
        self.parser.add_argument(
            "--initial_learning_rate",
            type=float,
            default=INITIAL_LEARNING_RATE)
        self.parser.add_argument(
            "--decay_steps",
            type=int,
            default=DECAY_STEPS)
        self.parser.add_argument(
            "--decay_factor",
            type=float,
            default=DECAY_FACTOR)

    def get_parser(
            self):
        return self.parser
