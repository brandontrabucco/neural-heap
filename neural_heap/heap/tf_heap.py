import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.heap.tf_heap_args import TFHeapArgs
from neural_heap.heap.tf_heap_utils import TFHeapUtils
from neural_heap.heap.tf_heap_utils import py_func
import tensorflow as tf
from tensorflow.python.framework import ops

class TFHeap(object):
    
    def __init__(
            self,
            parser=None):
        self.tf_heap_args = TFHeapArgs(parser=parser)
        self.tf_heap_utils = TFHeapUtils()
    
    def interact(
            self,
            args,
            operator,
            operand):
        with ops.name_scope(
                args.HEAP_NAME,
                args.HEAP_NAME,
                [operator, operand]) as name:
            result = py_func(
                self.tf_heap_utils.interact,
                [tf.cast(operator, tf.float64), 
                    tf.cast(operand, tf.float64)],
                [tf.float64],
                name=name,
                grad=self.tf_heap_utils.interact_grad)
            return tf.reshape(
                tf.cast(result[0], tf.float64), 
                [args.BATCH_SIZE, args.HEAP_SIZE])

    def reset( 
            self,
            args):
        with ops.name_scope(
                args.HEAP_NAME + "_reset",
                args.HEAP_NAME + "_reset",
                []) as name:
            result = py_func(
                self.tf_heap_utils.reset,
                [],
                [],
                name=name + "_reset",
                grad=self.tf_heap_utils.reset_grad)
            return result