import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
import tensorflow as tf

class TFGraphUtils(object):

    def _init__(
            self):
        pass

    def set_args(
            self,
            config):
        self.config = config

    def initialize_weights_cpu(
            self, 
            name,
            shape,
            standard_deviation=0.01,
            decay_factor=None,
            collection=None):
        with tf.device("/cpu:0"):
            weights = tf.get_variable(
                name,
                shape,
                initializer=tf.truncated_normal_initializer(
                    stddev=standard_deviation,
                    dtype=tf.float32),
                dtype=tf.float32)
        if decay_factor is not None and collection is not None:
            weight_decay = tf.multiply(
                tf.nn.l2_loss(weights),
                decay_factor,
                name=(name + self.config.EXTENSION_LOSS))
            tf.add_to_collection(collection, weight_decay)
        return weights

    def initialize_biases_cpu(
            self,
            name,
            shape):
        with tf.device("/cpu:0"):
            biases = tf.get_variable(
                name,
                shape,
                initializer=tf.constant_initializer(1.0),
                dtype=tf.float32)
        return biases

    def cross_entropy(
            self,
            prediction,
            labels,
            collection):
        entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=prediction))
        tf.add_to_collection(collection, entropy)
        return entropy

    def minimize(
            self, 
            loss,
            parameters):
        learning_rate = tf.train.exponential_decay(
            self.config.INITIAL_LEARNING_RATE,
            tf.get_collection(self.config.GLOBAL_STEP_OP)[0],
            self.config.DECAY_STEPS,
            self.config.DECAY_FACTOR,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradient = optimizer.minimize(
            loss, 
            var_list=parameters)
        return gradient