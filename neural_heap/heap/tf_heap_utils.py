import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset import BATCH_SIZE
from neural_heap.heap import HEAP_SIZE
from neural_heap.heap import OP_SIZE
import tensorflow as tf
import numpy as np
from heapq import heappush, heappop

# Credit to @harpone for the example
# https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(
        func,
        inp,
        Tout,
        stateful=True,
        name=None,
        grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class TFHeapNode(object):
    def __init__(self, x, outer):
        self.x = x
        self.outer = outer
    def __str__(self):
        return str(self.x)
    def __repr__(self):
        return repr(self.x)
    def __eq__(self, y):
        return self.outer.hash(self.x) == y.outer.hash(y.x)
    def __ne__(self, y):
        return self.outer.hash(self.x) != y.outer.hash(y.x)
    def __gt__(self, y):
        return self.outer.hash(self.x) > y.outer.hash(y.x)
    def __lt__(self, y):
        return self.outer.hash(self.x) < y.outer.hash(y.x)
    def __ge__(self, y):
        return self.outer.hash(self.x) >= y.outer.hash(y.x)
    def __le__(self, y):
        return self.outer.hash(self.x) <= y.outer.hash(y.x)

class TFHeapUtils(object):

    def __init__(
            self):
        pass

    def set_args(
            self,
            config):
        self.config = config
        self.hash_weights = np.arange(
            1, self.config.HEAP_SIZE + 1)
        self.reset()

    def hash(self, x):
        return np.sum(x * self.hash_weights)
        
    def __str__(self):
        return str(self.pq)
    
    def __repr__(self):
        return repr(self.pq)
        
    def interact(self, operator, operand):
        operator = operator.reshape(
            (self.config.BATCH_SIZE, self.config.OP_SIZE))
        operand = operand.reshape(
            (self.config.BATCH_SIZE, self.config.HEAP_SIZE))
        action = np.argmax(operator, axis=-1)
        result = []
        for i in range(self.config.BATCH_SIZE):
            if action[i] == 0: #push
                heappush(
                    self.pq[i], 
                    TFHeapNode(
                        operand[i, :], 
                        self))
                result += [operand[i, :]]
            elif action[i] == 1: #peek
                if len(self.pq[i]) > 0:
                    result += [self.pq[i][0].x]
                else:
                    result += [np.zeros(self.config.HEAP_SIZE)]
            else: #poll
                if len(self.pq[i]) > 0:
                    result += [heappop(self.pq[i]).x]
                else:
                    result += [np.zeros(self.config.HEAP_SIZE)]
        return np.vstack(result)
    
    def interact_grad(self, op, grad):
        operator = op.inputs[0]
        return np.zeros(operator.shape), grad

    def reset(self):
        self.pq = [[] for _ in range(self.config.BATCH_SIZE)]

    def reset_grad(self, op, grad):
        pass