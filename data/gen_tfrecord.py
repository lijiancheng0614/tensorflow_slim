"""Generate TFRecord."""

import os
import sys
import math
import argparse
import tensorflow as tf

sys.path.insert(0, '../')
from datasets import dataset_utils


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images_dir',
                        type=str,
                        default='images',
                        help='Images directory.')
    parser.add_argument('--list_train_path',
                        type=str,
                        default='list_train.txt',
                        help='List train path.')
    parser.add_argument('--list_val_path',
                        type=str,
                        default='list_val.txt',
                        help='List val path.')
    parser.add_argument('--output_train_dir',
                        type=str,
                        default='train/',
                        help='Output train tfrecord directory.')
    parser.add_argument('--output_val_dir',
                        type=str,
                        default='val/',
                        help='Output val tfrecord directory.')
    args = parser.parse_args()
    return args


def gen_dataset(list_path, images_dir, output_dir, num_shards=5):
    """Generate TFRecord.

    Args:
        list_path: str, list path.
        images_dir: str, images directory.
        output_dir: str, output directory.
        num_shards: int, number of shards.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lines = [line.split() for line in open(list_path)]
    num_per_shard = int(math.ceil(len(lines) / float(num_shards)))
    with tf.Graph().as_default():
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                output_path = os.path.join(
                    output_dir, 'data_{:05}-of-{:05}.tfrecord'.format(
                        shard_id, num_shards))
                tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
                start_index = shard_id * num_per_shard
                end_index = min((shard_id + 1) * num_per_shard, len(lines))
                for i in range(start_index, end_index):
                    sys.stdout.write(
                        '\r>> Converting image {}/{} shard {}'.format(
                            i + 1, len(lines), shard_id))
                    sys.stdout.flush()
                    image_data = tf.gfile.FastGFile(
                        os.path.join(images_dir, lines[i][0]), 'rb').read()
                    image = sess.run(decode_jpeg,
                                     feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(lines[i][1]))
                    tfrecord_writer.write(example.SerializeToString())
                tfrecord_writer.close()
                sys.stdout.write('\n')
                sys.stdout.flush()


def main():
    """main."""
    args = get_args()
    gen_dataset(args.list_train_path, args.images_dir, args.output_train_dir)
    gen_dataset(args.list_val_path, args.images_dir, args.output_val_dir)


if __name__ == '__main__':
    main()
