"""Generic generation script to prepare data."""

import os
import random
import argparse


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_dir',
                        type=str,
                        default='images',
                        help='Data directory.')
    parser.add_argument('--val_ratio',
                        type=float,
                        default=0.1,
                        help='The ratio of validation data.')
    parser.add_argument('--output_labels_path',
                        type=str,
                        default='labels.txt',
                        help='Labels path.')
    parser.add_argument('--output_list_path',
                        type=str,
                        default='list.txt',
                        help='List path.')
    parser.add_argument('--output_list_train_path',
                        type=str,
                        default='list_train.txt',
                        help='List train path.')
    parser.add_argument('--output_list_val_path',
                        type=str,
                        default='list_val.txt',
                        help='List val path.')
    args = parser.parse_args()
    return args


def main():
    """main."""
    args = get_args()
    # get class names
    class_names = sorted([
        _ for _ in os.listdir(args.images_dir)
        if os.path.isdir(os.path.join(args.images_dir, _))
    ])
    with open(args.output_labels_path, 'w') as fid:
        for class_name in class_names:
            fid.write('{}\n'.format(class_name))
    # get list
    lines = list()
    with open(args.output_list_path, 'w') as fid:
        for i, class_name in enumerate(class_names):
            images_list = os.listdir(os.path.join(args.images_dir, class_name))
            for image_name in images_list:
                line = '{}/{} {}\n'.format(class_name, image_name, i)
                fid.write(line)
                lines.append(line)
    # generate train/val list
    n_val = int(len(lines) * args.val_ratio)
    random.shuffle(lines)
    with open(args.output_list_val_path, 'w') as fid:
        for line in lines[:n_val]:
            fid.write(line)
    with open(args.output_list_train_path, 'w') as fid:
        for line in lines[n_val:]:
            fid.write(line)


if __name__ == '__main__':
    main()
