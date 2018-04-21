import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
import tensorflow as tf

class TFDatasetUtils(object):

    def _init__(
            self):
        pass

    def set_args(
            self,
            config):
        self.config = config

    def tokenize_example(self, example):
        return tf.one_hot(
            example,
            self.config.DATASET_RANGE,
            axis=-1,
            dtype=tf.float32)
        
    def detokenize_example(self, example):
        return tf.argmax(
            example,
            axis=-1)

    def decode_record(self, queue):
        reader = tf.TextLineReader()
        key, text = reader.read(queue)
        columns = tf.decode_csv(
            text,
            [[self.config.DATASET_DEFAULT] for i in range(2 * self.config.DATASET_COLUMNS)])
        x_inputs =  tf.stack(columns[:self.config.DATASET_COLUMNS])
        x_labels =  tf.stack(columns[self.config.DATASET_COLUMNS:])
        token_inputs = self.tokenize_example(x_inputs)
        token_labels = self.tokenize_example(x_labels)
        return token_inputs, token_labels

    def generate_batch(
            self, 
            token_inputs,
            token_labels,
            capacity,
            min_after_dequeue):
        inputs_batch, labels_batch = tf.train.shuffle_batch(
            [token_inputs,token_labels],
            batch_size=self.config.BATCH_SIZE,
            num_threads=self.config.NUM_THREADS,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return inputs_batch, labels_batch