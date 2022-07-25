import numpy as np
import argparse
import pickle
import os


def reshape_data(input_dir, output_dir) -> None:
    for split in ['train.pkl', 'test.pkl', 'validation.pkl']:
        input_path = os.path.join(input_dir, split)
        with open(input_path, 'rb') as f:
            source, target = pickle.load(f)
        # source = np.array(source)
        # target = np.array(target)
        expanded_sources = []
        expanded_targets = []
        for src, tgt in zip(source, target):
            new_src, new_tgt = shift_sequence(src, tgt)
            expanded_sources.append(new_src)
            expanded_targets.append(new_tgt)
        output_path = os.path.join(output_dir, split)
        pickle.dump((expanded_sources, expanded_targets), open(output_path, "wb"))


def shift_sequence(src: np.ndarray, tgt: np.ndarray):
    """

    Args:
        src:
        tgt:

    Returns:

    """
    all_src = []
    all_tgt = []
    src_len = src.shape[0]
    tgt_len = tgt.shape[0]
    full_sequence = np.vstack((src, tgt))
    for i in range(tgt_len):
        all_src.append(full_sequence[i: i+src_len])
        all_tgt.append(full_sequence[i+1: i+1+src_len])
    return all_src, all_tgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Location of preprocessed data',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to store output pickle data'
    )
    args = parser.parse_args()
    reshape_data(input_dir=args.input_dir, output_dir=args.output_dir)


