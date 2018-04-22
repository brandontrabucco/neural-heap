import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.experiment.tf_experiment import TFExperiment

if __name__ == "__main__":
    net = TFExperiment()
    for i in range(20):
        net.train(use_pretrained=(i > 0))
        net.test()