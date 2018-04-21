import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")
from neural_heap.graph.tf_graph_args import TFGraphArgs
from neural_heap.graph.tf_graph_utils import TFGraphUtils
from neural_heap.heap.tf_heap import TFHeap
from neural_heap.dataset.tf_dataset import TFDataset
import tensorflow as tf

class TFGraph(object):

    def __init__(
            self,
            parser=None):
        self.tf_graph_args = TFGraphArgs(parser=parser)
        self.tf_graph_utils = TFGraphUtils()
        self.tf_dataset = TFDataset(parser=parser)
        self.heap = TFHeap(parser=parser)

    def inference(
            self,
            args,
            x_batch):
        with tf.variable_scope(
                (args.PREFIX_CONTROLLER + args.EXTENSION_NUMBER(0))) as scope:
            lstm_forward = [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    args.LSTM_SIZE), 
                input_keep_prob=1.0, 
                output_keep_prob=args.DROPOUT_PROBABILITY) for i in range(args.LSTM_DEPTH)]
            lstm_forward = tf.contrib.rnn.MultiRNNCell(lstm_forward)
            state_batch = lstm_forward.zero_state(
                args.BATCH_SIZE,
                tf.float32)

            controller_w = self.tf_graph_utils.initialize_weights_cpu(
                (scope.name + args.PREFIX_OUTPUT + args.EXTENSION_WEIGHTS), 
                [args.LSTM_SIZE, args.DATASET_RANGE])
            controller_b = self.tf_graph_utils.initialize_biases_cpu(
                (scope.name + args.PREFIX_OUTPUT + args.EXTENSION_BIASES), 
                [args.DATASET_RANGE])

            operator_w = self.tf_graph_utils.initialize_weights_cpu(
                (scope.name + args.PREFIX_OPERATOR + args.EXTENSION_WEIGHTS), 
                [args.LSTM_SIZE, args.OP_SIZE])
            operator_b = self.tf_graph_utils.initialize_biases_cpu(
                (scope.name + args.PREFIX_OPERATOR + args.EXTENSION_BIASES), 
                [args.OP_SIZE])

            operand_w = self.tf_graph_utils.initialize_weights_cpu(
                (scope.name + args.PREFIX_OPERAND + args.EXTENSION_WEIGHTS), 
                [args.LSTM_SIZE, args.HEAP_SIZE])
            operand_b = self.tf_graph_utils.initialize_biases_cpu(
                (scope.name + args.PREFIX_OPERAND + args.EXTENSION_BIASES), 
                [args.HEAP_SIZE])
            heap_batch = tf.zeros([
                args.BATCH_SIZE, 
                args.HEAP_SIZE])

            inputs_batch = [
                tf.reshape(tf.slice(x_batch, [0, i, 0], [
                    args.BATCH_SIZE, 
                    1, 
                    args.DATASET_RANGE]),
                    [args.BATCH_SIZE, args.DATASET_RANGE])
                for i in range(args.DATASET_COLUMNS * 2)]
            outputs_batch = []

            for i in inputs_batch:
                x_inputs = tf.concat([
                    i, 
                    tf.cast(heap_batch, tf.float32)], 1)
                hidden_batch, state_batch = lstm_forward.call(
                    x_inputs,
                    state_batch)
                outputs_batch += [tf.add(tf.tensordot(
                    hidden_batch,
                    controller_w,
                    1), controller_b)]
                operator_batch = tf.add(tf.tensordot(
                    hidden_batch,
                    operator_w,
                    1), operator_b)
                operand_batch = tf.add(tf.tensordot(
                    hidden_batch,
                    operand_w,
                    1), operand_b)
                heap_batch = self.heap.interact(
                    args,
                    operator_batch,
                    operand_batch)
            
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)
            for p in parameters:
                tf.add_to_collection(
                    (args.PREFIX_CONTROLLER + args.COLLECTION_PARAMETERS),
                    p)
        return tf.reshape(
            tf.stack(outputs_batch, axis=1), [
                args.BATCH_SIZE,
                args.DATASET_COLUMNS * 2,
                args.DATASET_RANGE])

    def build_graph(
            self,
            args):
        self.tf_graph_utils.set_args(args)
        self.tf_dataset.tf_dataset_utils.set_args(args)
        self.heap.tf_heap_utils.set_args(args)
        global_step = tf.Variable(
            0,
            trainable=False,
            name=args.GLOBAL_STEP_OP)
        tf.add_to_collection(args.GLOBAL_STEP_OP, global_step)
        increment = tf.assign(
            global_step,
            global_step + 1)
        inputs_batch, labels_batch = self.tf_dataset.get_training_batch(args)
        tf.add_to_collection(args.INPUTS_OP, inputs_batch)
        tf.add_to_collection(args.LABELS_OP, labels_batch)

        tf.add_to_collection(
            args.INPUTS_DETOKENIZED_OP,
            self.tf_dataset.tf_dataset_utils.detokenize_example(inputs_batch))
        tf.add_to_collection(
            args.LABELS_DETOKENIZED_OP,
            self.tf_dataset.tf_dataset_utils.detokenize_example(labels_batch))
        prediction = self.inference(args, inputs_batch)
        tf.add_to_collection(args.PREDICTION_OP, prediction)
        tf.add_to_collection(
            args.PREDICTION_DETOKENIZED_OP,
            self.tf_dataset.tf_dataset_utils.detokenize_example(prediction))
            
        loss = self.tf_graph_utils.cross_entropy(
            prediction,
            labels_batch,
            (args.PREFIX_CONTROLLER + args.COLLECTION_LOSSES))
        tf.add_to_collection(args.LOSS_OP, loss)
        controller_parameters = tf.get_collection(
            args.PREFIX_CONTROLLER + args.COLLECTION_PARAMETERS)
        gradient = self.tf_graph_utils.minimize(loss, controller_parameters)
        reset_op = self.heap.reset(args)
        tf.add_to_collection(
            args.TRAIN_OP, 
            tf.group(gradient, increment, reset_op))
        tf.add_to_collection(
            args.VAL_OP, 
            tf.group(increment, reset_op))
        return controller_parameters

    def __call__(
            self,
            args=None):
        if args is None:
            args = self.tf_graph_args.get_parser().parse_args()
        self.tf_graph_utils.set_args(args)
        return self.build_graph(args)