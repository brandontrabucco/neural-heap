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

    def l2_loss(
            self,
            prediction,
            labels,
            collection):
        l2_norm = tf.nn.l2_loss(
                labels - prediction)
        tf.add_to_collection(collection, l2_norm)
        return l2_norm

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

    def argmax_state_tuple(
            self,
            state_buffer,
            best_q_enumerated):
        state_batch = []
        for z in range(self.config.LSTM_DEPTH):
            cell_state = tf.stack([
                state_buffer[a][z].c for a in range(self.config.OP_SIZE)
            ], axis=1)
            cell_state = tf.gather_nd(
                cell_state, 
                best_q_enumerated)
            hidden_state = tf.stack([
                state_buffer[a][z].h for a in range(self.config.OP_SIZE)
            ], axis=1)
            hidden_state = tf.gather_nd(
                hidden_state, 
                best_q_enumerated)
            lstm_tuple = tf.contrib.rnn.LSTMStateTuple(
                cell_state, 
                hidden_state)
            state_batch += [lstm_tuple]
        return tuple(state_batch)

    def expand_hidden_state(
            self,
            actions_batch,
            x_inputs,
            lstm_forward,
            state_batch,
            q_function_w,
            q_function_b):
        hidden_buffer = []
        state_buffer = []
        q_buffer = []
        for a in actions_batch:
            a_inputs = tf.concat([
                x_inputs,
                a], axis=1)
            q_hidden_batch, q_state_batch = lstm_forward.call(
                a_inputs,
                state_batch)
            hidden_buffer += [q_hidden_batch]
            state_buffer += [q_state_batch]
            q_buffer += [tf.add(tf.tensordot(
                q_hidden_batch,
                q_function_w,
                1), q_function_b)]

        best_q = tf.reduce_max(
            tf.stack(q_buffer, axis=1), 
            axis=1)
        best_q_indices = tf.argmax(
            tf.stack(q_buffer, axis=1), 
            axis=1, 
            output_type=tf.int32)
        best_q_enumerated = tf.stack([
                tf.range(self.config.BATCH_SIZE, dtype=tf.int32),
                best_q_indices], axis=1)
        return (hidden_buffer, 
            state_buffer,
            best_q, 
            best_q_enumerated, 
            best_q_indices)

    def prepare_inputs_actions(
            self, 
            x_batch):
        inputs_batch = [
            tf.reshape(tf.slice(x_batch, [0, i, 0], [
                self.config.BATCH_SIZE, 
                1, 
                self.config.DATASET_RANGE]),
                [self.config.BATCH_SIZE, self.config.DATASET_RANGE])
            for i in range(self.config.DATASET_COLUMNS * 2)]
        actions_batch = [
            tf.tile(
                tf.reshape(
                    tf.one_hot(i, self.config.OP_SIZE),
                    [1, self.config.OP_SIZE]),
                [self.config.BATCH_SIZE, 1]) for i in range(self.config.OP_SIZE)]
        return inputs_batch, actions_batch