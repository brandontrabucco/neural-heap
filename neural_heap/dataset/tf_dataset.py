import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.dataset.tf_dataset_args import TFDatasetArgs
from neural_heap.dataset.tf_dataset_utils import TFDatasetUtils
import tensorflow as tf

class TFDataset(object):

    def __init__(
            self,
            parser=None):
        self.tf_dataset_args = TFDatasetArgs(parser=parser)
        self.tf_dataset_utils = TFDatasetUtils()

    def get_training_batch(
            self,
            args):
        queue = tf.train.string_input_producer(
            args.DATASET_FILENAMES["train"])
        token_inputs, token_labels = self.tf_dataset_utils.decode_record(
            queue)
        inputs_batch, labels_batch = self.tf_dataset_utils.generate_batch(
            token_inputs,
            token_labels,
            args.TRAIN_EXAMPLES,
            args.TRAIN_EXAMPLES // args.MIN_RATIO)
        return (
            tf.concat([
                inputs_batch,
                tf.zeros([
                    args.BATCH_SIZE,
                    args.DATASET_COLUMNS,
                    args.DATASET_RANGE])],
                1),
            tf.concat([
                tf.zeros([
                    args.BATCH_SIZE,
                    args.DATASET_COLUMNS,
                    args.DATASET_RANGE]),
                labels_batch],
                1))

    def get_val_batch(
            self,
            args):
        queue = tf.train.string_input_producer(
            args.DATASET_FILENAMES["val"])
        token_inputs, token_labels = self.tf_dataset_utils.decode_record(
            queue)
        inputs_batch, labels_batch = self.tf_dataset_utils.generate_batch(
            token_inputs,
            token_labels,
            args.VAL_EXAMPLES,
            args.VAL_EXAMPLES // args.MIN_RATIO)
        return (
            tf.concat([
                inputs_batch,
                tf.zeros([
                    args.BATCH_SIZE,
                    args.DATASET_COLUMNS,
                    args.DATASET_RANGE])],
                1),
            tf.concat([
                tf.zeros([
                    args.BATCH_SIZE,
                    args.DATASET_COLUMNS,
                    args.DATASET_RANGE]),
                labels_batch],
                1))

    def __call__(
            self,
            args=None):
        if args is None:
            args = self.tf_dataset_args.get_parser().parse_args()
        self.tf_dataset_utils.set_args(args)
        train_batch = self.get_training_batch(args)
        val_batch = self.get_val_batch(args)
        return train_batch, val_batch