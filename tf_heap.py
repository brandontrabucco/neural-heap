import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from heapq import heappush, heappop

# Credit to @harpone for the example
# https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class TFMinHeapPQ(object):
    """This clas enables a min heap priority queue 
    data structure to interface with tensorflow 
    computational graphs."""
    
    class Node(object):
        """This class is used as a container for vectors
        that can be compared using a hashing function."""
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
    
    def __init__(self, n, i, v):
        """Initialize the data structure with name, 
        the number instances, the size of element vectors, 
        and the hashing function."""
        self.name = n
        self.instances = i
        self.hidden_size = v
        self.hash_weights = np.random.normal(
            0, 1, (1, self.hidden_size))
        self._reset()

    def hash(self, x):
        return np.sum(x * self.hash_weights)
        
        
    def __str__(self):
        return str(self.pq)
    
    def __repr__(self):
        return repr(self.pq)
        
    def _interact(self, operator, operand):
        """Given a vector of action probabilities, 
        and a vector argument, interact with the 
        priority queue."""
        operator = operator.reshape(
            (self.instances, 3))
        operand = operand.reshape(
            (self.instances, self.hidden_size))
        action = np.argmax(operator, axis=-1)
        result = []
        for i in range(self.instances):
            if action[i] == 0: #push
                heappush(
                    self.pq[i], 
                    TFMinHeapPQ.Node(
                        operand[i, :], 
                        self))
                result += [operand[i, :]]
            elif action[i] == 1: #peek
                if len(self.pq[i]) > 0:
                    result += [self.pq[i][0].x]
                else :
                    result += [np.zeros(self.hidden_size)]
            else: #poll
                if len(self.pq[i]) > 0:
                    result += [heappop(self.pq[i]).x]
                else :
                    result += [np.zeros(self.hidden_size)]
        return np.vstack(result)
    
    def _interact_grad(self, op, grad):
        """Currently this op is not differentiable, 
        but gradient is passed through input."""
        operator = op.inputs[0]
        operand = op.inputs[1]
        return np.zeros(operator.shape), grad

    def _reset(self):
        self.pq = [[] for _ in range(self.instances)]

    def _reset_grad(self, op, grad):
        pass
    
    def interact(self, operator, operand):
        """Connect the data structure to delayed computation, 
        as part of a computational graph."""
        with ops.name_scope(
                self.name,
                self.name,
                [operator, operand]) as name:
            result = py_func(
                self._interact,
                [tf.cast(operator, tf.float64), 
                    tf.cast(operand, tf.float64)],
                [tf.float64],
                name=name,
                grad=self._interact_grad)
            return tf.reshape(
                tf.cast(result[0], tf.float64), 
                [self.instances, self.hidden_size])

    def reset(self):
        """ Reset the priority queue by removing
         all the elements inside. """
        with ops.name_scope(
                self.name + "_reset",
                self.name + "_reset",
                []) as name:
            result = py_func(
                self._reset,
                [],
                [],
                name=name + "_reset",
                grad=self._reset_grad)
            return result