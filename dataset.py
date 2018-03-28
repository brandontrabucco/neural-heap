import numpy as np
import argparse as ap

def get_args_parser(parser=None):
    if parser is None:
        parser = ap.ArgumentParser(
            description="Generate a custom sorting dataset")
    parser.add_argument("--range", type=int, default=32)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--train_instances", type=int, default=10000)
    parser.add_argument("--val_instances", type=int, default=1000)
    return parser

def generate_indices(
        args_range,
        args_length,
        args_instances):
    x = np.concatenate([
        np.zeros(
            (args_instances, 1),
            dtype=int),
        np.random.randint(
            1,
            high=(args_range - 1),
            size=(args_instances, args_length - 2),
            dtype=int),
        np.full(
            (args_instances, 1),
            (args_range - 1),
            dtype=int)], axis=1)
    return x, np.sort(x, axis=1)

def get_one_hot(
        args_range,
        args_length,
        args_instances,
        indices):
    x = np.zeros((
        args_instances,
        args_length,
        args_range))
    for i in range(args_instances):
        for j in range(args_length):
            x[i, j, indices[i, j]] = 1.0
    return x

def get_dataset(
        args_range,
        args_length,
        args_instances):
    x, x_sorted = generate_indices(
        args_range,
        args_length,
        args_instances)
    return (x, get_one_hot(
        args_range,
        args_length,
        args_instances,
        x), x_sorted, get_one_hot(
            args_range,
            args_length,
            args_instances,
            x_sorted))

def get_train_dataset(args):
    return get_dataset(
        args.range,
        args.length,
        args.train_instances)

def get_val_dataset(args):
    return get_dataset(
        args.range,
        args.length,
        args.val_instances)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    train_dataset = get_train_dataset(args)
    np.savetxt("train_dataset.csv", np.concatenate([
        train_dataset[0],
        train_dataset[2]], axis=1), fmt="%1d", delimiter=",")
    val_dataset = get_val_dataset(args)
    np.savetxt("val_dataset.csv", np.concatenate([
        val_dataset[0],
        val_dataset[2]], axis=1), fmt="%1d", delimiter=",")