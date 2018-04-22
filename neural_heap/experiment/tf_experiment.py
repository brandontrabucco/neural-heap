import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.experiment.tf_config import TFConfig
from neural_heap.record.tf_train_record import TFTrainRecord
from neural_heap.record.tf_val_record import TFValRecord
from neural_heap.graph.tf_graph import TFGraph
import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

class TFExperiment(object):

    def __init__(
            self,
            parser=None):
        self.config = TFConfig()
        self.tf_graph = TFGraph(parser=parser)

    def train(
            self,
            use_pretrained=False):
        model_checkpoint = None
        with tf.Graph().as_default():
            if not use_pretrained:
                model_saver = tf.train.Saver(
                    var_list=(self.tf_graph(self.config) 
                        + tf.get_collection(self.config.GLOBAL_STEP_OP)))
                self.config(10, 0)
            else:
                model_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_BASEDIR)
                model_saver = tf.train.Saver(
                    var_list=(self.tf_graph(self.config) 
                        + tf.get_collection(self.config.GLOBAL_STEP_OP)))
                self.config(10, int(model_checkpoint.split("-")[1]))
            data_saver = TFTrainRecord(self.config)
            with tf.train.MonitoredTrainingSession(hooks=[
                    tf.train.StopAtStepHook(
                        num_steps=self.config.TRAIN_STOP_AT_STEP),
                    tf.train.CheckpointSaverHook(
                        self.config.CHECKPOINT_BASEDIR,
                        save_steps=self.config.TRAIN_ITERATIONS,
                        saver=model_saver),
                    data_saver]) as session:
                print("")
                print(datetime.now(), "Begin Training Experiment.")
                if model_checkpoint is not None:
                    model_saver.restore(session, model_checkpoint)
                else:
                    tf.train.global_step(
                        session,
                        tf.get_collection(self.config.GLOBAL_STEP_OP)[0])
                while not session.should_stop():
                    session.run(
                        tf.get_collection(self.config.TRAIN_OP))
                print(datetime.now(), "Finish Training Experiment.")
                print("")
            plt.plot(
                data_saver.iteration_points, 
                data_saver.loss_points,
                "b-o")
            plt.title("Controller Training")
            plt.xlabel("Batch Iteration")
            plt.ylabel("Mean Cross Entropy Loss")
            plt.grid(True)
            plt.savefig(
                self.config.PLOTS_BASEDIR + 
                datetime.now().strftime("%Y_%B_%d_%H_%M_%S") + 
                "_training_loss.png")
            plt.close()

    def test(
            self):
        model_checkpoint = tf.train.latest_checkpoint(self.config.CHECKPOINT_BASEDIR)
        self.config(1, int(model_checkpoint.split("-")[1]))
        with tf.Graph().as_default():
            model_saver = tf.train.Saver(
                    var_list=(self.tf_graph(self.config) 
                        + tf.get_collection(self.config.GLOBAL_STEP_OP)))
            data_saver = TFValRecord(self.config)
            with tf.train.MonitoredTrainingSession(hooks=[
                    tf.train.StopAtStepHook(num_steps=self.config.VAL_STOP_AT_STEP),
                    data_saver]) as session:
                print("")
                print(datetime.now(), "Begin Testing Experiment.")
                model_saver.restore(session, model_checkpoint)
                while not session.should_stop():
                    session.run(
                        tf.get_collection(self.config.VAL_OP))
                accuracy = (data_saver.correct_elements
                    / (self.config.VAL_EXAMPLES * self.config.DATASET_COLUMNS))
                print(
                    datetime.now(),
                    "Val Accuracy: %.2f %%" % (100 * accuracy))
                print(datetime.now(), "Finish Testing Experiment.")
                print("")
            return accuracy