import tensorflow as tf
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from datetime import datetime


class TrainRecordHook(tf.train.SessionRunHook):

    def __init__(
            self,
            config):
        self.config = config

    def begin(self):
        self.start_time = time()
        self.iteration_points = []
        self.loss_points = []

    def before_run(
            self,
            run_context):
        return tf.train.SessionRunArgs([
            tf.get_collection(self.config.GLOBAL_STEP_OP)[0],
            tf.get_collection(self.config.LOSS_OP)[0]])

    def after_run(
            self,
            run_context,
            run_values):
        current_step, current_loss = run_values.results
        current_time = time()
        batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
        self.start_time = current_time
        if current_step % max(self.config.TRAIN_ITERATIONS // self.config.TOTAL_LOGS, 1) == 0:
            print(
                datetime.now(),
                "Speed: %.2f" % batch_speed,
                "ETA: %.2f" % ((self.config.TRAIN_STOP_AT_STEP - current_step) 
                    / batch_speed / 60 / 60),
                "Iteration: %d" % current_step, 
                "Loss: %.2f" % current_loss)
            self.iteration_points.append(current_step)
            self.loss_points.append(current_loss)


class ValRecordHook(tf.train.SessionRunHook):

    def __init__(
            self,
            config):
        self.config = config

    def begin(self):
        self.start_time = time()
        self.correct_elements = 0

    def before_run(
            self,
            run_context):
        return tf.train.SessionRunArgs([
            tf.get_collection(self.config.LABELS_DETOKENIZED_OP)[0],
            tf.get_collection(self.config.PREDICTION_DETOKENIZED_OP)[0]])

    def after_run(
            self,
            run_context,
            run_values):
        current_labels, current_prediction = run_values.results
        current_labels = current_labels[self.config.DATASET_COLUMNS:]
        current_prediction = current_prediction[self.config.DATASET_COLUMNS:]
        current_time = time()
        batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
        self.start_time = current_time
        for i, j in zip(
                current_labels.flatten().tolist(),
                current_prediction.flatten().tolist()):
            if i == j:
                self.correct_elements += 1


class Configuration(object):

    def __init__(self):
        self.DATASET_FILENAMES = {
            "train": ["train_dataset.csv"],
            "val": ["val_dataset.csv"]}
        self.TRAIN_EXAMPLES = 10000
        self.VAL_EXAMPLES = 1000
        self.MIN_RATIO = 10
        self.DATASET_COLUMNS = 7
        self.DATASET_RANGE = 32
        self.DATASET_DEFAULT = self.DATASET_RANGE // 2
        self.BATCH_SIZE = 32
        self.TRAIN_EPOCH_SIZE = self.TRAIN_EXAMPLES // self.BATCH_SIZE
        self.VAL_EPOCH_SIZE = self.VAL_EXAMPLES // self.BATCH_SIZE
        self.NUM_THREADS = 2
        self.TOTAL_LOGS = 20
        self.CHECKPOINT_BASEDIR = "G:/My Drive/Academic/Research/Neural Heap/saves"
        self.PREFIX_TOTAL = "total"
        self.PREFIX_CONTROLLER = "controller"
        self.EXTENSION_NUMBER = (lambda number: "_" + str(number))
        self.EXTENSION_LOSS = "_loss"
        self.EXTENSION_WEIGHTS = "_weights"
        self.EXTENSION_BIASES = "_biases"
        self.COLLECTION_LOSSES = "_losses"
        self.COLLECTION_PARAMETERS = "_parameters"
        self.COLLECTION_ACTIVATIONS = "_activations"
        self.LOSS_OP = "loss_op"
        self.GLOBAL_STEP_OP = "global_step"
        self.PREDICTION_OP = "prediction_op"
        self.INPUTS_OP = "inputs_op"
        self.LABELS_OP = "labels_op"
        self.PREDICTION_DETOKENIZED_OP = "prediction_d_op"
        self.INPUTS_DETOKENIZED_OP = "inputs_d_op"
        self.LABELS_DETOKENIZED_OP = "labels_d_op"
        self.TRAIN_OP = "train_op"
        self.INCREMENT_OP = "increment_op"
        self.ENSEMBLE_SIZE = 4
        self.LSTM_SIZE = (self.DATASET_RANGE * 2 * self.ENSEMBLE_SIZE)
        self.LSTM_DEPTH = 2
        self.DROPOUT_PROBABILITY = (1 / self.ENSEMBLE_SIZE)
        self.USE_DROPOUT = True
        self.INITIAL_LEARNING_RATE = 0.001
        self.DECAY_STEPS = self.TRAIN_EPOCH_SIZE
        self.DECAY_FACTOR = 0.8

    def __call__(self, num_epoch, offset):
        self.NUM_EPOCH = num_epoch
        self.TRAIN_ITERATIONS = (self.TRAIN_EPOCH_SIZE * self.NUM_EPOCH)
        self.TRAIN_STOP_AT_STEP = self.TRAIN_ITERATIONS + offset
        self.VAL_ITERATIONS = self.VAL_EPOCH_SIZE
        self.VAL_STOP_AT_STEP = self.VAL_ITERATIONS + offset


class Experiment(object):

    def __init__(self):
        self.config = Configuration()

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

    def get_training_batch(self):
        queue = tf.train.string_input_producer(
            self.config.DATASET_FILENAMES["train"])
        token_inputs, token_labels = self.decode_record(
            queue)
        inputs_batch, labels_batch = self.generate_batch(
            token_inputs,
            token_labels,
            self.config.TRAIN_EXAMPLES,
            self.config.TRAIN_EXAMPLES // self.config.MIN_RATIO)
        return (
            tf.concat([
                inputs_batch,
                tf.zeros([
                    self.config.BATCH_SIZE,
                    self.config.DATASET_COLUMNS,
                    self.config.DATASET_RANGE])],
                1),
            tf.concat([
                tf.zeros([
                    self.config.BATCH_SIZE,
                    self.config.DATASET_COLUMNS,
                    self.config.DATASET_RANGE]),
                labels_batch],
                1))

    def get_val_batch(self):
        queue = tf.train.string_input_producer(
            self.config.DATASET_FILENAMES["val"])
        token_inputs, token_labels = self.decode_record(
            queue)
        inputs_batch, labels_batch = self.generate_batch(
            token_inputs,
            token_labels,
            self.config.VAL_EXAMPLES,
            self.config.VAL_EXAMPLES // self.config.MIN_RATIO)
        return (
            tf.concat([
                inputs_batch,
                tf.zeros([
                    self.config.BATCH_SIZE,
                    self.config.DATASET_COLUMNS,
                    self.config.DATASET_RANGE])],
                1),
            tf.concat([
                tf.zeros([
                    self.config.BATCH_SIZE,
                    self.config.DATASET_COLUMNS,
                    self.config.DATASET_RANGE]),
                labels_batch],
                1))

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

    def inference(self, x_batch):
        with tf.variable_scope(
                (self.config.PREFIX_CONTROLLER + self.config.EXTENSION_NUMBER(1))) as scope:
            lstm_forward = [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(self.config.LSTM_SIZE), 
                input_keep_prob=1.0, 
                output_keep_prob=self.config.DROPOUT_PROBABILITY) for i in range(self.config.LSTM_DEPTH)]
            lstm_backward = [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(self.config.LSTM_SIZE), 
                input_keep_prob=1.0, 
                output_keep_prob=self.config.DROPOUT_PROBABILITY) for i in range(self.config.LSTM_DEPTH)]
            lstm_forward = tf.contrib.rnn.MultiRNNCell(lstm_forward)
            lstm_backward = tf.contrib.rnn.MultiRNNCell(lstm_backward)
            output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
                lstm_forward,
                lstm_backward,
                x_batch,
                initial_state_fw=lstm_forward.zero_state(
                    self.config.BATCH_SIZE,
                    tf.float32),
                initial_state_bw=lstm_forward.zero_state(
                    self.config.BATCH_SIZE,
                    tf.float32),
                dtype=tf.float32)
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)
            for p in parameters:
                tf.add_to_collection(
                    (self.config.PREFIX_CONTROLLER + self.config.COLLECTION_PARAMETERS),
                    p)
        with tf.variable_scope(
                (self.config.PREFIX_CONTROLLER + self.config.EXTENSION_NUMBER(1))) as scope:
            linear_w = self.initialize_weights_cpu(
                (scope.name + self.config.EXTENSION_WEIGHTS), 
                [self.config.LSTM_SIZE * 2, self.config.DATASET_RANGE])
            linear_b = self.initialize_biases_cpu(
                (scope.name + self.config.EXTENSION_BIASES), 
                [self.config.DATASET_RANGE])
            output_batch = tf.add(tf.tensordot(
                tf.concat(output_batch, 2),
                linear_w,
                1), linear_b)
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)
            for p in parameters:
                tf.add_to_collection(
                    (self.config.PREFIX_CONTROLLER + self.config.COLLECTION_PARAMETERS),
                    p)
        return tf.reshape(
            output_batch, [
                self.config.BATCH_SIZE,
                self.config.DATASET_COLUMNS * 2,
                self.config.DATASET_RANGE])

    def cross_entropy(
            self,
            prediction,
            labels,
            collection):
        entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=prediction,
                dim=-1))
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

    def build_graph(self):
        global_step = tf.Variable(
            0,
            trainable=False,
            name=self.config.GLOBAL_STEP_OP)
        tf.add_to_collection(self.config.GLOBAL_STEP_OP, global_step)
        increment = tf.assign(
            global_step,
            global_step + 1)
        tf.add_to_collection(self.config.INCREMENT_OP, increment)
        inputs_batch, labels_batch = self.get_training_batch()
        tf.add_to_collection(self.config.INPUTS_OP, inputs_batch)
        tf.add_to_collection(self.config.LABELS_OP, labels_batch)
        tf.add_to_collection(
            self.config.INPUTS_DETOKENIZED_OP,
            self.detokenize_example(inputs_batch))
        tf.add_to_collection(
            self.config.LABELS_DETOKENIZED_OP,
            self.detokenize_example(labels_batch))
        prediction = self.inference(inputs_batch)
        tf.add_to_collection(self.config.PREDICTION_OP, prediction)
        tf.add_to_collection(
            self.config.PREDICTION_DETOKENIZED_OP,
            self.detokenize_example(prediction))
        loss = self.cross_entropy(
            prediction,
            labels_batch,
            (self.config.PREFIX_CONTROLLER + self.config.COLLECTION_LOSSES))
        tf.add_to_collection(self.config.LOSS_OP, loss)
        controller_parameters = tf.get_collection(
            self.config.PREFIX_CONTROLLER + self.config.COLLECTION_PARAMETERS)
        gradient = self.minimize(loss, controller_parameters)
        tf.add_to_collection(self.config.TRAIN_OP, gradient)
        return controller_parameters

    def train(
            self,
            model_checkpoint=None):
        with tf.Graph().as_default():
            if model_checkpoint is None:
                model_saver = tf.train.Saver(
                    var_list=(self.build_graph() 
                        + tf.get_collection(self.config.GLOBAL_STEP_OP)))
                self.config(1, 0)
            else:
                model_saver = tf.train.import_meta_graph(
                    model_checkpoint + ".meta")
                self.config(1, int(model_checkpoint.split("-")[1]))
            data_saver = TrainRecordHook(self.config)
            with tf.train.MonitoredTrainingSession(hooks=[
                    tf.train.StopAtStepHook(
                        num_steps=self.config.TRAIN_STOP_AT_STEP),
                    tf.train.CheckpointSaverHook(
                        self.config.CHECKPOINT_BASEDIR,
                        save_steps=self.config.TRAIN_STOP_AT_STEP,
                        saver=model_saver),
                    data_saver]) as session:
                if model_checkpoint is not None:
                    model_saver.restore(session, model_checkpoint)
                else:
                    tf.train.global_step(
                        session,
                        tf.get_collection(self.config.GLOBAL_STEP_OP)[0])
                while not session.should_stop():
                    session.run([
                        tf.get_collection(self.config.INCREMENT_OP)[0],
                        tf.get_collection(self.config.TRAIN_OP)[0]])
            plt.plot(
                data_saver.iteration_points, 
                data_saver.loss_points,
                "b-o")
            plt.title("Controller Training")
            plt.xlabel("Batch Iteration")
            plt.ylabel("Mean Cross Entropy Loss")
            plt.grid(True)
            plt.savefig(
                datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
                "_training_loss.png")
            plt.close()

    def test(self):
        model_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_BASEDIR)
        self.config(1, int(model_checkpoint.split("-")[1]))
        with tf.Graph().as_default():
            model_saver = tf.train.import_meta_graph(
                model_checkpoint + ".meta")
            data_saver = ValRecordHook(self.config)
            with tf.train.MonitoredTrainingSession(hooks=[
                    tf.train.StopAtStepHook(num_steps=self.config.VAL_STOP_AT_STEP),
                    data_saver]) as session:
                model_saver.restore(session, model_checkpoint)
                while not session.should_stop():
                    session.run([
                        tf.get_collection(self.config.INCREMENT_OP)[0],
                        tf.get_collection(self.config.LABELS_DETOKENIZED_OP)[0],
                        tf.get_collection(self.config.PREDICTION_DETOKENIZED_OP)[0]])
            return (data_saver.correct_elements
                / (self.config.VAL_EXAMPLES * self.config.DATASET_COLUMNS * 2))


if __name__ == "__main__":
    net = Experiment()
    net.train()
    print(
        datetime.now(),
        "Val Accuracy: %.2f %%" % (100 * net.test()))