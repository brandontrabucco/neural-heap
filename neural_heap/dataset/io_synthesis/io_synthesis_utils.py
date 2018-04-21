import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
import numpy

class IOSynthesisUtils(object):

    def __init__(self):
        pass

    def generate_indices(
            self,
            args_range,
            args_length,
            args_instances):
        x = numpy.concatenate([
            numpy.zeros(
                (args_instances, 1),
                dtype=int),
            numpy.random.randint(
                1,
                high=(args_range - 1),
                size=(args_instances, args_length - 2),
                dtype=int),
            numpy.full(
                (args_instances, 1),
                (args_range - 1),
                dtype=int)], axis=1)
        return x, numpy.sort(x, axis=1)

    def get_one_hot(
            self,
            args_range,
            args_length,
            args_instances,
            indices):
        x = numpy.zeros((
            args_instances,
            args_length,
            args_range))
        for i in range(args_instances):
            for j in range(args_length):
                x[i, j, indices[i, j]] = 1.0
        return x

    def get_dataset(
            self,
            args_range,
            args_length,
            args_instances):
        x, x_sorted = self.generate_indices(
            args_range,
            args_length,
            args_instances)
        return (x, self.get_one_hot(
            args_range,
            args_length,
            args_instances,
            x), x_sorted, self.get_one_hot(
                args_range,
                args_length,
                args_instances,
                x_sorted))