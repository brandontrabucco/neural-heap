import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.heap import HEAP_SIZE
from neural_heap.heap import OP_SIZE
import argparse

class TFHeapArgs(object):

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
            "--heap_size",
            type=int,
            default=HEAP_SIZE)
        self.parser.add_argument(
            "--op_size",
            type=int,
            default=OP_SIZE)

    def get_parser(
            self):
        return self.parser
