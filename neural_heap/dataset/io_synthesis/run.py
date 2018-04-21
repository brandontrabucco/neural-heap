import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
sys.path.insert(0, os.getcwd())
from neural_heap.dataset.io_synthesis.io_synthesis import IOSynthesis
import numpy

if __name__ == "__main__":

    io_synthesis = IOSynthesis()
    train_dataset, val_dataset = io_synthesis()

    numpy.savetxt(
        "neural_heap/dataset/csv/train_dataset.csv", 
        numpy.concatenate([
            train_dataset[0],
            train_dataset[2]], axis=1), fmt="%1d", delimiter=",")
    numpy.savetxt(
        "neural_heap/dataset/csv/val_dataset.csv", 
        numpy.concatenate([
            val_dataset[0],
            val_dataset[2]], axis=1), fmt="%1d", delimiter=",")