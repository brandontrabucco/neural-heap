import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime

class TFValRecord(tf.train.SessionRunHook):

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
        current_labels = current_labels[:, self.config.DATASET_COLUMNS:]
        current_prediction = current_prediction[:, self.config.DATASET_COLUMNS:]
        current_time = time()
        batch_speed = 1.0 / (current_time - self.start_time + 1e-3)
        self.start_time = current_time
        for i, j in zip(
                current_labels.flatten().tolist(),
                current_prediction.flatten().tolist()):
            if i == j:
                self.correct_elements += 1