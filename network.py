import tensorflow as tf
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from datetime import datetime

DATASET_FILENAMES = {
    "train": ["train_dataset.csv"],
    "val": ["val_dataset.csv"]}
TRAIN_EXAMPLES = 10000
VAL_EXAMPLES = 1000
MIN_RATIO = 10
DATASET_COLUMNS = 7
DATASET_RANGE = 32
DATASET_DEFAULT = DATASET_RANGE // 2
BATCH_SIZE = 32
TRAIN_EPOCH_SIZE = TRAIN_EXAMPLES // BATCH_SIZE
VAL_EPOCH_SIZE = VAL_EXAMPLES // BATCH_SIZE
NUM_THREADS = 2
TOTAL_LOGS = 20
CHECKPOINT_BASEDIR = "G:/My Drive/Academic/Research/Neural Heap/saves"
PREFIX_TOTAL = "total"
PREFIX_CONTROLLER = "controller"
EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
COLLECTION_LOSSES = "_losses"
COLLECTION_PARAMETERS = "_parameters"
COLLECTION_ACTIVATIONS = "_activations"
LOSS_OP = "loss_op"
GLOBAL_STEP_OP = "global_step"
PREDICTION_OP = "prediction_op"
INPUTS_OP = "inputs_op"
LABELS_OP = "labels_op"
PREDICTION_DETOKENIZED_OP = "prediction_d_op"
INPUTS_DETOKENIZED_OP = "inputs_d_op"
LABELS_DETOKENIZED_OP = "labels_d_op"
TRAIN_OP = "train_op"
INCREMENT_OP = "increment_op"
ENSEMBLE_SIZE = 1
LSTM_SIZE = (DATASET_RANGE * 2 * ENSEMBLE_SIZE)
DROPOUT_PROBABILITY = (1 / ENSEMBLE_SIZE)
USE_DROPOUT = False
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = TRAIN_EPOCH_SIZE
DECAY_FACTOR = 0.5

def tokenize_example(example):
    return tf.one_hot(
        example,
        DATASET_RANGE,
        axis=-1,
        dtype=tf.float32)
    
def detokenize_example(example):
    return tf.argmax(
        example,
        axis=-1)

def decode_record(queue):
    reader = tf.TextLineReader()
    key, text = reader.read(queue)
    columns = tf.decode_csv(
        text,
        [[DATASET_DEFAULT] for i in range(2 * DATASET_COLUMNS)])
    x_inputs =  tf.stack(columns[:DATASET_COLUMNS])
    x_labels =  tf.stack(columns[DATASET_COLUMNS:])
    token_inputs = tokenize_example(x_inputs)
    token_labels = tokenize_example(x_labels)
    return token_inputs, token_labels

def generate_batch(token_inputs, token_labels, capacity, min_after_dequeue):
    inputs_batch, labels_batch = tf.train.shuffle_batch(
        [token_inputs,token_labels],
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return inputs_batch, labels_batch

def get_training_batch():
    queue = tf.train.string_input_producer(
        DATASET_FILENAMES["train"])
    token_inputs, token_labels = decode_record(
        queue)
    inputs_batch, labels_batch = generate_batch(
        token_inputs,
        token_labels,
        TRAIN_EXAMPLES,
        TRAIN_EXAMPLES // MIN_RATIO)
    return (
        tf.concat([
            inputs_batch,
            tf.zeros([BATCH_SIZE, DATASET_COLUMNS, DATASET_RANGE])],
            1),
        tf.concat([
            tf.zeros([BATCH_SIZE, DATASET_COLUMNS, DATASET_RANGE]),
            labels_batch],
            1))

def get_val_batch():
    queue = tf.train.string_input_producer(
        DATASET_FILENAMES["val"])
    token_inputs, token_labels = decode_record(
        queue)
    inputs_batch, labels_batch = generate_batch(
        token_inputs,
        token_labels,
        VAL_EXAMPLES,
        VAL_EXAMPLES // MIN_RATIO)
    return (
        tf.concat([
            inputs_batch,
            tf.zeros([BATCH_SIZE, DATASET_COLUMNS, DATASET_RANGE])],
            1),
        tf.concat([
            tf.zeros([BATCH_SIZE, DATASET_COLUMNS, DATASET_RANGE]),
            labels_batch],
            1))

def initialize_weights_cpu(
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
            name=(name + EXTENSION_LOSS))
        tf.add_to_collection(collection, weight_decay)
    return weights

def initialize_biases_cpu(name, shape):
    with tf.device("/cpu:0"):
        biases = tf.get_variable(
            name,
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)
    return biases

def inference(x_batch):
    global CONTROLLER_INITIALIZED
    with tf.variable_scope(
            (PREFIX_CONTROLLER + EXTENSION_NUMBER(1)),
            reuse=CONTROLLER_INITIALIZED) as scope:
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=1.0, 
                output_keep_prob=DROPOUT_PROBABILITY)
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            x_batch,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32)
        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            for p in parameters:
                tf.add_to_collection((PREFIX_CONTROLLER + COLLECTION_PARAMETERS), p)

    with tf.variable_scope(
            (PREFIX_CONTROLLER + EXTENSION_NUMBER(2)),
            reuse=CONTROLLER_INITIALIZED) as scope:
        lstm_forward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        lstm_backward = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        initial_state_fw = lstm_forward.zero_state(BATCH_SIZE, tf.float32)
        initial_state_bw = lstm_backward.zero_state(BATCH_SIZE, tf.float32)
        if USE_DROPOUT:
            lstm_forward = tf.contrib.rnn.DropoutWrapper(
                lstm_forward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)
            lstm_backward = tf.contrib.rnn.DropoutWrapper(
                lstm_backward, 
                input_keep_prob=DROPOUT_PROBABILITY, 
                output_keep_prob=DROPOUT_PROBABILITY)
        output_batch, state_batch = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            tf.concat(output_batch, 2),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32)
        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            for p in parameters:
                tf.add_to_collection((PREFIX_CONTROLLER + COLLECTION_PARAMETERS), p)

    with tf.variable_scope(
            (PREFIX_CONTROLLER + EXTENSION_NUMBER(3)),
                reuse=CONTROLLER_INITIALIZED) as scope:
        linear_w = initialize_weights_cpu(
            (scope.name + EXTENSION_WEIGHTS), 
            [LSTM_SIZE*2, DATASET_RANGE])
        linear_b = initialize_biases_cpu(
            (scope.name + EXTENSION_BIASES), 
            [DATASET_RANGE])
        output_batch = tf.add(tf.tensordot(tf.concat(output_batch, 2), linear_w, 1), linear_b)
        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            for p in parameters:
                tf.add_to_collection((PREFIX_CONTROLLER + COLLECTION_PARAMETERS), p)

    CONTROLLER_INITIALIZED = True
    return tf.reshape(
        output_batch,
        [BATCH_SIZE, DATASET_COLUMNS * 2, DATASET_RANGE])

def reset_kernel():
    global CONTROLLER_INITIALIZED
    CONTROLLER_INITIALIZED = False

def cross_entropy(prediction, labels, collection):
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=prediction,
            dim=-1))
    tf.add_to_collection(collection, entropy)
    return entropy

def minimize(loss, parameters):
    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        tf.get_collection(GLOBAL_STEP_OP)[0],
        DECAY_STEPS,
        DECAY_FACTOR,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient = optimizer.minimize(
        loss, 
        var_list=parameters)
    return gradient

def build_graph():
    global_step = tf.Variable(
        0,
        trainable=False,
        name=GLOBAL_STEP_OP)
    tf.add_to_collection(GLOBAL_STEP_OP, global_step)
    increment = tf.assign(
        global_step,
        global_step + 1)
    tf.add_to_collection(INCREMENT_OP, increment)
    inputs_batch, labels_batch = get_training_batch()
    tf.add_to_collection(INPUTS_OP, inputs_batch)
    tf.add_to_collection(LABELS_OP, labels_batch)
    tf.add_to_collection(
        INPUTS_DETOKENIZED_OP,
        detokenize_example(inputs_batch))
    tf.add_to_collection(
        LABELS_DETOKENIZED_OP,
        detokenize_example(labels_batch))
    prediction = inference(inputs_batch)
    tf.add_to_collection(PREDICTION_OP, prediction)
    tf.add_to_collection(
        PREDICTION_DETOKENIZED_OP,
        detokenize_example(prediction))
    loss = cross_entropy(
        prediction,
        labels_batch,
        (PREFIX_CONTROLLER + COLLECTION_LOSSES))
    tf.add_to_collection(LOSS_OP, loss)
    controller_parameters = tf.get_collection(
        PREFIX_CONTROLLER + COLLECTION_PARAMETERS)
    gradient = minimize(loss, controller_parameters)
    tf.add_to_collection(TRAIN_OP, gradient)
    return controller_parameters

def train(num_epochs=1, model_checkpoint=None):
    step_delta = (TRAIN_EPOCH_SIZE 
        * num_epochs)
    num_steps = step_delta
    reset_kernel()
    with tf.Graph().as_default():

        class DataSaver(tf.train.SessionRunHook):
            def begin(self):
                self.start_time = time()
                self.iteration_points = []
                self.loss_points = []
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([
                    tf.get_collection(GLOBAL_STEP_OP)[0],
                    tf.get_collection(LOSS_OP)[0]])
            def after_run(self, run_context, run_values):
                current_step, current_loss = run_values.results
                current_time = time()
                batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
                self.start_time = current_time
                if current_step % max(step_delta // TOTAL_LOGS, 1) == 0:
                    print(
                        datetime.now(),
                        "Speed: %.2f" % batch_speed,
                        "ETA: %.2f" % ((num_steps - current_step) / batch_speed / 60 / 60),
                        "Iteration: %d" % current_step, 
                        "Loss: %.2f" % current_loss)
                    self.iteration_points.append(current_step)
                    self.loss_points.append(current_loss)

        if model_checkpoint is None:
            model_saver = tf.train.Saver(
                var_list=(build_graph() 
                    + tf.get_collection(GLOBAL_STEP_OP)))
        else:
            model_saver = tf.train.import_meta_graph(
                model_checkpoint + ".meta")
            num_steps += int(model_checkpoint.split("-")[1])
        data_saver = DataSaver()
        with tf.train.MonitoredTrainingSession(hooks=[
                tf.train.StopAtStepHook(
                    num_steps=num_steps),
                tf.train.CheckpointSaverHook(
                    CHECKPOINT_BASEDIR,
                    save_steps=num_steps,
                    saver=model_saver),
                data_saver]) as session:
            if model_checkpoint is not None:
                model_saver.restore(session, model_checkpoint)
            else:
                tf.train.global_step(
                    session,
                    tf.get_collection(GLOBAL_STEP_OP)[0])
            while not session.should_stop():
                session.run([
                    tf.get_collection(INCREMENT_OP)[0],
                    tf.get_collection(TRAIN_OP)[0]])

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

def test():
    model_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_BASEDIR)
    num_steps = (VAL_EPOCH_SIZE 
        + int(model_checkpoint.split("-")[1]))
    with tf.Graph().as_default():
        class DataSaver(tf.train.SessionRunHook):
            def begin(self):
                self.start_time = time()
                self.correct_elements = 0
            def before_run(self, run_context):
                return tf.train.SessionRunArgs([
                    tf.get_collection(LABELS_DETOKENIZED_OP)[0],
                    tf.get_collection(PREDICTION_DETOKENIZED_OP)[0]])
            def after_run(self, run_context, run_values):
                current_labels, current_prediction = run_values.results
                current_time = time()
                batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
                self.start_time = current_time
                for i, j in zip(
                        current_labels.flatten().tolist(),
                        current_prediction.flatten().tolist()):
                    if i == j:
                        self.correct_elements += 1

        model_saver = tf.train.import_meta_graph(
                model_checkpoint + ".meta")
        data_saver = DataSaver()
        with tf.train.MonitoredTrainingSession(hooks=[
                tf.train.StopAtStepHook(num_steps=num_steps),
                data_saver]) as session:
            model_saver.restore(session, model_checkpoint)
            while not session.should_stop():
                session.run([
                    tf.get_collection(INCREMENT_OP)[0],
                    tf.get_collection(LABELS_DETOKENIZED_OP)[0],
                    tf.get_collection(PREDICTION_DETOKENIZED_OP)[0]])
        return (data_saver.correct_elements
            / (VAL_EXAMPLES * DATASET_COLUMNS * 2))

if __name__ == "__main__":
    train()
    for i in range(10):
        train(
            num_epochs=10,
            model_checkpoint=tf.train.latest_checkpoint(CHECKPOINT_BASEDIR))
        print(
            datetime.now(),
            "Val Accuracy: %.2f %%" % (100 * test()))