import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset.io_synthesis.io_synthesis_args import IOSynthesisArgs
from neural_heap.dataset.io_synthesis.io_synthesis_utils import IOSynthesisUtils

class IOSynthesis(object):

    def __init__(
            self,
            parser=None):
        self.io_synthesis_args = IOSynthesisArgs(parser=parser)
        self.io_synthesis_utils = IOSynthesisUtils()

    def get_train_dataset(
            self, 
            args):
        return self.io_synthesis_utils.get_dataset(
            args.range,
            args.length,
            args.train_instances)

    def get_val_dataset(
            self,
            args):
        return self.io_synthesis_utils.get_dataset(
            args.range,
            args.length,
            args.val_instances)

    def __call__(
            self,
            args=None):
        if args is None:
            args = self.io_synthesis_args.get_parser().parse_args()
        train_dataset = self.get_train_dataset(
            args)
        val_dataset = self.get_val_dataset(
            args)
        return train_dataset, val_dataset
        