import os, sys
os.chdir("G:\\My Drive\\Academic\\Research\\Neural Heap")

CHECKPOINT_BASEDIR = "G:/My Drive/Academic/Research/Neural Heap/saves/"
PLOTS_BASEDIR = "G:/My Drive/Academic/Research/Neural Heap/plots/"
PREFIX_TOTAL = "total"
PREFIX_CONTROLLER = "controller"
PREFIX_OUTPUT = "output"
PREFIX_OPERATOR = "operator"
PREFIX_OPERAND = "operand"
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
VAL_OP = "val_op"