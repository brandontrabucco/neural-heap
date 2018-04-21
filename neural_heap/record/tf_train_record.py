import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime

class TFTrainRecord(tf.train.SessionRunHook):

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
            tf.get_collection(self.config.LOSS_OP)[0],
            tf.get_collection(self.config.LABELS_DETOKENIZED_OP)[0],
            tf.get_collection(self.config.PREDICTION_DETOKENIZED_OP)[0]])

    def after_run(
            self,
            run_context,
            run_values):
        current_step, current_loss, current_labels, current_prediction = run_values.results
        current_time = time()
        batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
        self.start_time = current_time
        if current_step % max(self.config.TRAIN_ITERATIONS // self.config.TOTAL_LOGS, 1) == 0:
            correct_elements = 0
            current_labels = current_labels[:, self.config.DATASET_COLUMNS:]
            current_prediction = current_prediction[:, self.config.DATASET_COLUMNS:]
            print("Label:", current_labels[0], "Prediction:", current_prediction[0])
            for i, j in zip(
                current_labels.flatten().tolist(),
                current_prediction.flatten().tolist()):
                if i == j:
                    correct_elements += 1
            print(
                datetime.now(),
                "Speed: %.2f" % batch_speed,
                "ETA: %.2f" % ((self.config.TRAIN_STOP_AT_STEP - current_step) 
                    / batch_speed / 60 / 60),
                "Iteration: %d" % current_step, 
                "Loss: %.2f" % current_loss,
                "Train Accuracy: %.2f %%" % (100 * correct_elements 
                    / self.config.BATCH_SIZE 
                    / self.config.DATASET_COLUMNS))
            self.iteration_points.append(current_step)
            self.loss_points.append(current_loss)