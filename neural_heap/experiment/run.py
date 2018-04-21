import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.experiment.tf_experiment import TFExperiment

if __name__ == "__main__":
    net = TFExperiment()
    net.train()
    net.test()
    net.train(use_pretrained=True)
    net.test()